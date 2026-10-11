# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

# `Hadoop` is ported from Arrow C++'s `Lz4HadoopCodec`
# (`cpp/src/arrow/util/compression_lz4.cc`), Apache-2.0; see NOTICE.txt.

"""Parquet's page compression: `Compression`, the codec a column chunk
names, run by marrow's own compressors or, with `native` off, by the system
libraries' (`marrow.compress`); and `Hadoop`, the framing of
Parquet's deprecated `LZ4` codec."""

from ..buffers import Buffer
from ..compress import (
    Brotli,
    Compressor,
    LibBrotli,
    LibGzip,
    LibLz4,
    LibSnappy,
    LibZstd,
    Lz4,
    Snappy,
    Zstd,
    compress_with,
    decompress_with,
)
from ..utils.byteorder import BigEndian, LittleEndian
from ..errors import CorruptError, DynError, NotImplementedError
from ..views import BufferView


struct Hadoop[C: Compressor](Compressor):
    """Blocks of `C` in Hadoop frames -- a big-endian u32 decompressed size,
    a big-endian u32 compressed size, then the block -- as Arrow C++ writes
    Parquet's deprecated `LZ4` codec, one frame a page. Decoding takes a run
    of frames or, when the bytes do not parse as one, a plain block: what
    Parquet C++ wrote before it framed them."""

    @staticmethod
    def max_decompressed_length(n: Int) -> Int:
        return Self.C.max_decompressed_length(n)

    @staticmethod
    def compress(
        src: Span[mut=False, UInt8, _], mut dst: List[UInt8]
    ) raises DynError:
        var at = len(dst)
        LittleEndian.append[DType.uint64](dst, 0)  # both sizes, below
        Self.C.compress(src, dst)
        BigEndian.write[DType.uint32](dst, at, UInt32(len(src)))
        BigEndian.write[DType.uint32](dst, at + 4, UInt32(len(dst) - at - 8))

    @staticmethod
    def decompress[
        o: Origin[mut=True]
    ](src: Span[UInt8, _], dst: Span[UInt8, o]) raises DynError:
        if not Self._frames_into(src, dst):
            Self.C.decompress(src, dst)

    @staticmethod
    def _frames_into[
        o: Origin[mut=True]
    ](src: Span[UInt8, _], dst: Span[UInt8, o]) -> Bool:
        """Whether `src` is a run of frames that decodes to exactly `dst` --
        Arrow C++'s `TryDecompressHadoop`. A frame here is one block;
        Hadoop's own codec may split one across several, which no Parquet
        writer does."""
        var ip = 0
        var op = 0
        while len(src) - ip >= 8:
            var raw = Int(BigEndian.fixed[DType.uint32](src, ip))
            var size = Int(BigEndian.fixed[DType.uint32](src, ip + 4))
            ip += 8
            if size > len(src) - ip or raw > len(dst) - op:
                return False
            try:
                Self.C.decompress(src[ip : ip + size], dst[op : op + raw])
            except:
                return False
            ip += size
            op += raw
        return ip == len(src) and op == len(dst)


@fieldwise_init
struct Compression(Equatable, ImplicitlyCopyable, Movable):
    """A Parquet `CompressionCodec` value. Each codec runs in Mojo or, with
    `native` off, through its system library; GZIP always runs through
    zlib, and BROTLI is always written by libbrotli.

    Enum values:
        0 UNCOMPRESSED  1 SNAPPY  2 GZIP  4 BROTLI  5 LZ4  6 ZSTD  7 LZ4_RAW
    """

    var code: Int

    comptime UNCOMPRESSED = Self(0)
    comptime SNAPPY = Self(1)
    comptime GZIP = Self(2)
    comptime BROTLI = Self(4)
    comptime LZ4 = Self(5)
    comptime ZSTD = Self(6)
    comptime LZ4_RAW = Self(7)

    def needs_libs(self, native: Bool) -> Bool:
        """Whether decompressing this codec may call into a `dlopen`ed
        library: all but UNCOMPRESSED and -- when `native` -- the codecs in
        Mojo, BROTLI, LZ4, LZ4_RAW, SNAPPY and ZSTD. A codec not named here
        answers True, so one wired up later opens the libraries first rather
        than racing to."""
        if self == Self.UNCOMPRESSED:
            return False
        elif (
            self == Self.LZ4
            or self == Self.LZ4_RAW
            or self == Self.SNAPPY
            or self == Self.ZSTD
            or self == Self.BROTLI
        ):
            return not native
        else:
            return True

    def max_decompressed_length(self, n: Int) -> Int:
        """The most `n` bytes of this codec decode to."""
        if self == Self.UNCOMPRESSED:
            return n
        elif self == Self.ZSTD:
            return Zstd.max_decompressed_length(n)
        elif self == Self.SNAPPY:
            return Snappy.max_decompressed_length(n)
        elif self == Self.LZ4 or self == Self.LZ4_RAW:
            return Lz4.max_decompressed_length(n)
        elif self == Self.GZIP:
            return LibGzip.max_decompressed_length(n)
        else:
            return Brotli.max_decompressed_length(n)

    def decompress_into(
        self,
        native: Bool,
        src: Span[UInt8, _],
        out_size: Int,
        mut dst: List[UInt8],
    ) raises:
        """Append `src` decompressed, `out_size` bytes, to `dst` -- a page's
        reused scratch, after its levels for a v2 page."""
        self._check_size(src, out_size)
        var at = len(dst)
        dst.resize(unsafe_uninit_length=at + out_size)
        self._decompress_to(native, src, Span(dst)[at:])

    def decompress_owned(
        self, native: Bool, src: Span[UInt8, _], out_size: Int
    ) raises -> Buffer[mut=False]:
        """Decompress `src` into a buffer of its own, for a decoder that keeps
        pointing into the page after the reader has moved on."""
        self._check_size(src, out_size)
        var out = Buffer.alloc_uninit[DType.uint8](out_size)
        self._decompress_to(
            native, src, out.view[DType.uint8]().as_span()[:out_size]
        )
        return out^.to_immutable()

    def _check_size(self, src: Span[UInt8, _], out_size: Int) raises:
        """`out_size` comes from the page header: refuse one `src` cannot
        decode to before anything is allocated for it."""
        if out_size < 0 or out_size > self.max_decompressed_length(len(src)):
            raise CorruptError(
                t"parquet: {len(src)} compressed bytes cannot hold {out_size}"
            )

    def _decompress_to[
        o: Origin[mut=True]
    ](self, native: Bool, src: Span[UInt8, _], dst: Span[UInt8, o]) raises:
        """Decompress `src` into exactly `dst`."""
        if self == Self.UNCOMPRESSED:
            BufferView(dst).copy_from(src[: len(dst)])
        elif self == Self.ZSTD:
            decompress_with[Zstd, LibZstd](native, src, dst)
        elif self == Self.SNAPPY:
            decompress_with[Snappy, LibSnappy](native, src, dst)
        elif self == Self.LZ4_RAW:
            decompress_with[Lz4, LibLz4](native, src, dst)
        elif self == Self.LZ4:
            decompress_with[Hadoop[Lz4], Hadoop[LibLz4]](native, src, dst)
        elif self == Self.GZIP:
            LibGzip.decompress(src, dst)
        elif self == Self.BROTLI:
            decompress_with[Brotli, LibBrotli](native, src, dst)
        else:
            raise NotImplementedError(
                t"parquet: unsupported compression codec {self.code}"
            )

    def compress(self, native: Bool, src: Span[UInt8, _]) raises -> List[UInt8]:
        """Compress `src`, returning the codec's output bytes."""
        var out = List[UInt8]()
        if self == Self.UNCOMPRESSED:
            out.extend(src)
        elif self == Self.ZSTD:
            compress_with[Zstd, LibZstd](native, src, out)
        elif self == Self.SNAPPY:
            compress_with[Snappy, LibSnappy](native, src, out)
        elif self == Self.LZ4_RAW:
            compress_with[Lz4, LibLz4](native, src, out)
        elif self == Self.LZ4:
            compress_with[Hadoop[Lz4], Hadoop[LibLz4]](native, src, out)
        elif self == Self.GZIP:
            LibGzip.compress(src, out)
        elif self == Self.BROTLI:
            LibBrotli.compress(src, out)
        else:
            raise NotImplementedError(
                t"parquet: unsupported compression codec {self.code}"
            )
        return out^
