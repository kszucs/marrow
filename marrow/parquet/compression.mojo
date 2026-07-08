"""Page compression codecs via runtime FFI to the system C libraries.

Parquet compresses each page body with one of a handful of codecs. Rather than
reimplement them, we `dlopen` the standard C libraries (`libzstd`, `libsnappy`,
`liblz4`, `libz`) at runtime and call their block APIs — the same approach
arrow-rs/duckdb take, just without a link-time dependency. Handles are opened
lazily and cached in a `Codecs` value that lives for one read/write.

The uncompressed page size is always known from the page header, so decompress
never has to probe the frame for an output size.

Parquet `CompressionCodec` enum:
    0 UNCOMPRESSED  1 SNAPPY  2 GZIP  3 LZO  4 BROTLI  5 LZ4  6 ZSTD  7 LZ4_RAW
"""

from std.ffi import OwnedDLHandle, _try_find_dylib
from std.pathlib import Path
from std.memory import alloc, memset_zero

comptime CODEC_UNCOMPRESSED: Int = 0
comptime CODEC_SNAPPY: Int = 1
comptime CODEC_GZIP: Int = 2
comptime CODEC_BROTLI: Int = 4
comptime CODEC_LZ4: Int = 5
comptime CODEC_ZSTD: Int = 6
comptime CODEC_LZ4_RAW: Int = 7

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


struct Codecs(Movable):
    """Lazily-opened handles to the compression libraries, cached per file. Each
    codec's decode logic lives in its own `_*_decompress` method so `decompress`
    stays a thin dispatch."""

    var _zstd: Optional[OwnedDLHandle]
    var _snappy: Optional[OwnedDLHandle]
    var _lz4: Optional[OwnedDLHandle]
    var _zlib: Optional[OwnedDLHandle]

    def __init__(out self):
        self._zstd = None
        self._snappy = None
        self._lz4 = None
        self._zlib = None

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

    # --- per-codec decoders (write exactly `out_size` bytes to `dst`) ---

    def _zstd_decompress(
        mut self,
        src: Span[UInt8, _],
        dst: UnsafePointer[UInt8, _],
        out_size: Int,
    ) raises:
        self._ensure_zstd()
        var n = self._zstd.value().call["ZSTD_decompress", Int](
            dst, out_size, src.unsafe_ptr(), len(src)
        )
        if n != out_size:
            raise Error("zstd: decompressed size mismatch")

    def _snappy_decompress(
        mut self,
        src: Span[UInt8, _],
        dst: UnsafePointer[UInt8, _],
        out_size: Int,
    ) raises:
        self._ensure_snappy()
        var sz = alloc[UInt](1)
        sz[0] = UInt(out_size)
        var status = self._snappy.value().call["snappy_uncompress", Int32](
            src.unsafe_ptr(), len(src), dst, sz
        )
        var produced = Int(sz[0])
        sz.free()
        if status != 0 or produced != out_size:
            raise Error("snappy: decompress failed")

    def _lz4_raw_decompress(
        mut self,
        src: Span[UInt8, _],
        dst: UnsafePointer[UInt8, _],
        out_size: Int,
    ) raises:
        self._ensure_lz4()
        var n = self._lz4.value().call["LZ4_decompress_safe", Int32](
            src.unsafe_ptr(), dst, Int32(len(src)), Int32(out_size)
        )
        if Int(n) != out_size:
            raise Error("lz4: decompressed size mismatch")

    def _gzip_decompress(
        mut self,
        src: Span[UInt8, _],
        dst: UnsafePointer[UInt8, _],
        out_size: Int,
    ) raises:
        self._ensure_zlib()
        ref z = self._zlib.value()
        # z_stream is 112 bytes on LP64; drive it directly. Fields we set:
        # next_in @0, avail_in @8, next_out @24, avail_out @32; total_out @40.
        var strm = alloc[UInt64](16)
        var sp = strm.bitcast[UInt8]()
        memset_zero(sp, 128)
        strm[0] = UInt64(Int(src.unsafe_ptr()))
        (sp + 8).bitcast[UInt32]()[0] = UInt32(len(src))
        strm[3] = UInt64(Int(dst))
        (sp + 32).bitcast[UInt32]()[0] = UInt32(out_size)

        var version = z.call[
            "zlibVersion", UnsafePointer[UInt8, MutUntrackedOrigin]
        ]()
        # windowBits 31 = 15 | 16 → gzip; stream_size = sizeof(z_stream) = 112
        var rc = z.call["inflateInit2_", Int32](
            sp, Int32(31), version, Int32(112)
        )
        if Int(rc) != 0:
            strm.free()
            raise Error("gzip: inflateInit2 failed")
        var st = z.call["inflate", Int32](sp, Int32(4))  # Z_FINISH
        var produced = Int(strm[5])  # total_out
        _ = z.call["inflateEnd", Int32](sp)
        strm.free()
        if Int(st) != 1:  # Z_STREAM_END
            raise Error("gzip: inflate failed")
        if produced != out_size:
            raise Error("gzip: decompressed size mismatch")

    def decompress(
        mut self, codec: Int, src: Span[UInt8, _], out_size: Int
    ) raises -> List[UInt8]:
        """Decompress `src` into a fresh `out_size`-byte list."""
        if codec == CODEC_UNCOMPRESSED:
            var out = List[UInt8]()
            out.extend(src)
            return out^

        var dst = List[UInt8]()
        dst.resize(unsafe_uninit_length=out_size)  # codec fills every byte
        var ptr = dst.unsafe_ptr()
        if codec == CODEC_ZSTD:
            self._zstd_decompress(src, ptr, out_size)
        elif codec == CODEC_SNAPPY:
            self._snappy_decompress(src, ptr, out_size)
        elif codec == CODEC_LZ4_RAW:
            self._lz4_raw_decompress(src, ptr, out_size)
        elif codec == CODEC_GZIP:
            self._gzip_decompress(src, ptr, out_size)
        else:
            raise Error(
                "parquet: unsupported compression codec " + String(codec)
            )
        return dst^

    def compress(
        mut self, codec: Int, src: Span[UInt8, _]
    ) raises -> List[UInt8]:
        """Compress `src`, returning the codec's output bytes. Writers currently
        emit UNCOMPRESSED, SNAPPY, or ZSTD."""
        if codec == CODEC_UNCOMPRESSED:
            var out = List[UInt8]()
            out.extend(src)
            return out^

        var src_ptr = src.unsafe_ptr()
        if codec == CODEC_ZSTD:
            self._ensure_zstd()
            var bound = self._zstd.value().call["ZSTD_compressBound", Int](
                len(src)
            )
            var dst = alloc[UInt8](bound)
            var n = self._zstd.value().call["ZSTD_compress", Int](
                dst, bound, src_ptr, len(src), Int32(1)
            )
            var out = List[UInt8]()
            for i in range(n):
                out.append(dst[i])
            dst.free()
            return out^
        elif codec == CODEC_SNAPPY:
            self._ensure_snappy()
            var bound = self._snappy.value().call[
                "snappy_max_compressed_length", Int
            ](len(src))
            var dst = alloc[UInt8](bound)
            var sz = alloc[UInt](1)
            sz[0] = UInt(bound)
            _ = self._snappy.value().call["snappy_compress", Int32](
                src_ptr, len(src), dst, sz
            )
            var produced = Int(sz[0])
            var out = List[UInt8]()
            for i in range(produced):
                out.append(dst[i])
            dst.free()
            sz.free()
            return out^
        else:
            raise Error(
                "parquet: unsupported compression codec " + String(codec)
            )
