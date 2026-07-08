"""Page compression codecs via runtime FFI to the system C libraries.

Parquet compresses each page body with one of a handful of codecs. Rather than
reimplement them, we `dlopen` the standard C libraries (`libzstd`, `libsnappy`)
at runtime and call their block APIs — the same approach arrow-rs/duckdb take,
just without a link-time dependency. Libraries are opened lazily and cached in a
`Codecs` value that lives for the duration of one read/write.

The uncompressed page size is always known from the page header, so decompress
never has to probe the frame for an output size.

Parquet `CompressionCodec` enum:
    0 UNCOMPRESSED  1 SNAPPY  2 GZIP  3 LZO  4 BROTLI  5 LZ4  6 ZSTD  7 LZ4_RAW
"""

from std.ffi import OwnedDLHandle, _try_find_dylib
from std.pathlib import Path
from std.memory import alloc

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


struct Codecs(Movable):
    """Lazily-opened handles to the compression libraries, cached per file."""

    var _zstd: Optional[OwnedDLHandle]
    var _snappy: Optional[OwnedDLHandle]

    def __init__(out self):
        self._zstd = None
        self._snappy = None

    def _ensure_zstd(mut self) raises:
        if not self._zstd:
            self._zstd = _try_find_dylib["zstd"](materialize[_ZSTD_PATHS]())

    def _ensure_snappy(mut self) raises:
        if not self._snappy:
            self._snappy = _try_find_dylib["snappy"](
                materialize[_SNAPPY_PATHS]()
            )

    def decompress(
        mut self, codec: Int, src: Span[UInt8, _], out_size: Int
    ) raises -> List[UInt8]:
        """Decompress `src` into a fresh `out_size`-byte list."""
        if codec == CODEC_UNCOMPRESSED:
            var out = List[UInt8]()
            out.extend(src)
            return out^

        var dst = List[UInt8]()
        dst.resize(out_size, 0)
        var dst_ptr = dst.unsafe_ptr()
        var src_ptr = src.unsafe_ptr()

        if codec == CODEC_ZSTD:
            self._ensure_zstd()
            var n = self._zstd.value().call["ZSTD_decompress", Int](
                dst_ptr, out_size, src_ptr, len(src)
            )
            if n != out_size:
                raise Error("zstd: decompressed size mismatch")
        elif codec == CODEC_SNAPPY:
            self._ensure_snappy()
            var sz = alloc[UInt](1)
            sz[0] = UInt(out_size)
            var status = self._snappy.value().call["snappy_uncompress", Int32](
                src_ptr, len(src), dst_ptr, sz
            )
            var produced = Int(sz[0])
            sz.free()
            if status != 0 or produced != out_size:
                raise Error("snappy: decompress failed")
        else:
            raise Error(
                "parquet: unsupported compression codec " + String(codec)
            )

        return dst^

    def compress(
        mut self, codec: Int, src: Span[UInt8, _]
    ) raises -> List[UInt8]:
        """Compress `src`, returning the codec's output bytes."""
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
