# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""System-library loading for the block compression codecs.

The standard C libraries (`libsnappy`, `libz`, `libbrotli`) are `dlopen`-ed
at runtime and their block APIs called directly -- the same approach arrow-rs
and duckdb take, just without a link-time dependency. `CompressionLibs` is the
primitive block calls plus the per-call scratch they need; the handles
themselves live in the `Codecs` set below, one process-global for all six.

LZ4 and Zstandard run in Mojo by default, `marrow.utils.lz4` and
`marrow.utils.zstd`; a reader or writer created with `native_codecs=False`
runs them through `liblz4` and `libzstd` instead, the calls below, and the
tests check the Mojo codecs against the same libraries. Snappy also has a
Mojo implementation, `marrow.utils.snappy`, which Parquet does not use yet.

**Nothing here is Parquet-specific**, which is why it lives in `marrow.utils`
rather than in `marrow.parquet` where it started (as a second module named
`utils`). The format-specific half -- the Parquet `CompressionCodec` codes --
is `Compression` in `marrow.parquet.codecs`, which dispatches onto this.
"""

from ..errors import CorruptError, InternalError, InvalidError
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
    still holds one of these per worker -- and which implementation runs
    LZ4 and Zstandard."""

    var _sz: List[UInt]  # reusable size out-param for snappy
    var native: Bool
    """Whether LZ4 and Zstandard run in Mojo, the default, or through
    `liblz4` and `libzstd` -- `lz4_*` and `zstd_*` below."""

    def __init__(out self, native: Bool = True):
        self._sz = [UInt(0)]
        self.native = native

    @staticmethod
    def preload() raises:
        """Open every codec now, on the calling thread.

        `_Global` vends its pointer without locking and says nothing about
        racing *creation*, so the first touch must not be several workers at
        once. `ParquetFile.read` calls this before dispatching whenever a
        chunk it is about to decode needs a library
        (`Compression.needs_libs`); after it returns, every worker's
        `Codecs.handle[...]()` is a pure read.

        All six open together, because the caller knows only that
        *something* ahead needs one, not which. A missing one is not an
        error here -- each `Dylib` records its own failure and re-raises it
        at the call that needs it.
        """
        Codecs.preload()

    # --- decompress: write exactly `out_size` bytes to `dst` ---

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
            raise CorruptError("snappy: decompress failed")

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
            raise InternalError("gzip: inflateInit2 failed")
        var st = z.call["inflate", Int32](sp, Int32(4))  # Z_FINISH
        var produced = Int(strm[unsafe_offset=5])  # total_out
        _ = z.call["inflateEnd", Int32](sp)
        strm.unsafe_free()
        if Int(st) != 1:  # Z_STREAM_END
            raise CorruptError("gzip: inflate failed")
        if produced != out_size:
            raise CorruptError("gzip: decompressed size mismatch")

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
            raise CorruptError("brotli: decompress failed")

    # --- LZ4 and Zstandard through their libraries, when not `native` ---

    @staticmethod
    def zstd_decompress[
        o: Origin[mut=True]
    ](src: Span[UInt8, _], dst: Span[UInt8, o]) raises:
        """`ZSTD_decompress`: the frames in `src` into exactly `dst`."""
        var n = Codecs.handle["zstd"]().call["ZSTD_decompress", Int](
            dst.unsafe_ptr(), len(dst), src.unsafe_ptr(), len(src)
        )
        if n != len(dst):
            raise CorruptError(
                "zstd: libzstd could not decode the expected size"
            )

    @staticmethod
    def zstd_compress(src: Span[UInt8, _], mut dst: List[UInt8]) raises:
        """Append `ZSTD_compress`'s level-1 frame for `src` to `dst` -- what
        `Zstd.compress` writes, with libzstd 1.5.7 byte for byte."""
        var z = Codecs.handle["zstd"]()
        var bound = z.call["ZSTD_compressBound", Int](len(src))
        var at = len(dst)
        dst.resize(unsafe_uninit_length=at + bound)
        var n = z.call["ZSTD_compress", Int](
            dst.unsafe_ptr().unsafe_offset(at),
            bound,
            src.unsafe_ptr(),
            len(src),
            Int32(1),
        )
        if z.call["ZSTD_isError", UInt32](n) != 0:
            dst.shrink(at)
            raise InternalError("zstd: libzstd could not compress")
        dst.shrink(at + n)

    @staticmethod
    def lz4_decompress_block[
        o: Origin[mut=True]
    ](src: Span[UInt8, _], dst: Span[UInt8, o]) raises:
        """`LZ4_decompress_safe`: the block `src` into exactly `dst`."""
        if len(src) > Self._LZ4_MAX_INPUT or len(dst) > Int(Int32.MAX):
            raise InvalidError("lz4: liblz4 decodes blocks of under 2 GiB")
        var n = Codecs.handle["lz4"]().call["LZ4_decompress_safe", Int32](
            src.unsafe_ptr(), dst.unsafe_ptr(), Int32(len(src)), Int32(len(dst))
        )
        if Int(n) != len(dst):
            raise CorruptError("lz4: liblz4 could not decode the expected size")

    @staticmethod
    def lz4_compress_block(src: Span[UInt8, _], mut dst: List[UInt8]) raises:
        """Append `LZ4_compress_default`'s block for `src` to `dst` -- what
        `Lz4.compress_block` writes, with liblz4 1.10.0 byte for byte."""
        if len(src) > Self._LZ4_MAX_INPUT:
            raise InvalidError(
                t"lz4: {len(src)} bytes is over the block format's"
                t" {Self._LZ4_MAX_INPUT}"
            )
        var l = Codecs.handle["lz4"]()
        var bound = Int(l.call["LZ4_compressBound", Int32](Int32(len(src))))
        var at = len(dst)
        dst.resize(unsafe_uninit_length=at + bound)
        var n = l.call["LZ4_compress_default", Int32](
            src.unsafe_ptr(),
            dst.unsafe_ptr().unsafe_offset(at),
            Int32(len(src)),
            Int32(bound),
        )
        if n <= 0:
            dst.shrink(at)
            raise InternalError("lz4: liblz4 could not compress")
        dst.shrink(at + Int(n))

    @staticmethod
    def lz4_decompress_frame[
        o: Origin[mut=True]
    ](src: Span[UInt8, _], dst: Span[UInt8, o]) raises:
        """`LZ4F_decompress`: the one frame `src` into exactly `dst`, called
        until it reports the frame's end."""
        var l = Codecs.handle["lz4"]()
        var ctx = unsafe_alloc[Int](1)
        var sizes = unsafe_alloc[UInt](2)  # out capacity, in available
        # LZ4F_VERSION
        var rc = l.call["LZ4F_createDecompressionContext", Int](
            ctx, UInt32(100)
        )
        if l.call["LZ4F_isError", UInt32](rc) != 0:
            ctx.unsafe_free()
            sizes.unsafe_free()
            raise InternalError("lz4: liblz4 could not make a frame decoder")
        var dctx = ctx[unsafe_offset=0]
        var ip = 0
        var op = 0
        var done = False
        var failed = False
        while not done and not failed:
            sizes[unsafe_offset=0] = UInt(len(dst) - op)
            sizes[unsafe_offset=1] = UInt(len(src) - ip)
            var hint = l.call["LZ4F_decompress", Int](
                dctx,
                dst.unsafe_ptr().unsafe_offset(op),
                sizes,
                src.unsafe_ptr().unsafe_offset(ip),
                sizes.unsafe_offset(1),
                0,
            )
            var wrote = Int(sizes[unsafe_offset=0])
            var read = Int(sizes[unsafe_offset=1])
            op += wrote
            ip += read
            # An error, or no progress short of the end: the input ran out.
            failed = l.call["LZ4F_isError", UInt32](hint) != 0 or (
                hint != 0 and wrote == 0 and read == 0
            )
            done = hint == 0
        _ = l.call["LZ4F_freeDecompressionContext", Int](dctx)
        ctx.unsafe_free()
        sizes.unsafe_free()
        if failed or ip != len(src) or op != len(dst):
            raise CorruptError(
                "lz4: liblz4 could not decode one frame of the expected size"
            )

    @staticmethod
    def lz4_compress_frame(src: Span[UInt8, _], mut dst: List[UInt8]) raises:
        """Append `LZ4F_compressFrame`'s frame for `src`, with the default
        preferences Arrow C++ passes, to `dst` -- what `Lz4.compress_frame`
        writes, with liblz4 1.10.0 byte for byte."""
        var l = Codecs.handle["lz4"]()
        var bound = l.call["LZ4F_compressFrameBound", Int](len(src), 0)
        var at = len(dst)
        dst.resize(unsafe_uninit_length=at + bound)
        var n = l.call["LZ4F_compressFrame", Int](
            dst.unsafe_ptr().unsafe_offset(at),
            bound,
            src.unsafe_ptr(),
            len(src),
            0,
        )
        if l.call["LZ4F_isError", UInt32](n) != 0:
            dst.shrink(at)
            raise InternalError("lz4: liblz4 could not compress a frame")
        dst.shrink(at + n)

    comptime _LZ4_MAX_INPUT = 0x7E000000
    """liblz4's `LZ4_MAX_INPUT_SIZE`."""

    # --- compress: return the codec's output bytes ---

    @staticmethod
    def _take(dst: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> List[UInt8]:
        """Copy `n` bytes out of a freshly-`alloc`'d compression scratch buffer,
        free the buffer, and return an owned List — the shared tail of every
        `*_compress` method."""
        var out = c_bytes(dst, n)
        dst.unsafe_free()
        return out^

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
            raise InternalError("gzip: deflateInit2 failed")
        var st = z.call["deflate", Int32](sp, Int32(4))  # Z_FINISH
        var produced = Int(strm[unsafe_offset=5])  # total_out @40
        _ = z.call["deflateEnd", Int32](sp)
        strm.unsafe_free()
        if Int(st) != 1:  # Z_STREAM_END
            dst.unsafe_free()
            raise InternalError("gzip: deflate failed")
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
            raise InternalError("brotli: compression failed")
        return Self._take(dst, produced)
