# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Parquet's page compression: `Compression`, the codec a column chunk
names, mapped to the `Method` that runs it (`marrow.compress`)."""

from ...buffers import Buffer
from ...compress import Method
from ...errors import CorruptError, NotImplementedError


@fieldwise_init
struct Compression(Equatable, ImplicitlyCopyable, Movable):
    """A Parquet `CompressionCodec` value. Each codec runs in Mojo or, with
    `native` off, through its system library; GZIP always runs through
    zlib. BROTLI is written at quality 1 natively, quality 11 by libbrotli.

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
        library, as its method answers (`Method.needs_libs`). A codec with
        no method answers True."""
        try:
            return self.method().needs_libs(native)
        except:
            return True

    def method(self) raises NotImplementedError -> Method:
        """The method this codec's pages are compressed with."""
        if self == Self.UNCOMPRESSED:
            return Method.NONE
        elif self == Self.SNAPPY:
            return Method.SNAPPY
        elif self == Self.GZIP:
            return Method.GZIP
        elif self == Self.BROTLI:
            return Method.BROTLI
        elif self == Self.LZ4:
            return Method.LZ4_HADOOP
        elif self == Self.ZSTD:
            return Method.ZSTD
        elif self == Self.LZ4_RAW:
            return Method.LZ4
        raise NotImplementedError(
            t"parquet: unsupported compression codec {self.code}"
        )

    def decompress_into(
        self,
        native: Bool,
        src: Span[UInt8, _],
        out_size: Int,
        mut dst: List[UInt8],
    ) raises:
        """Append `src` decompressed, `out_size` bytes, to `dst` -- a page's
        reused scratch, after its levels for a v2 page."""
        var method = self._check_size(src, out_size)
        var at = len(dst)
        dst.resize(unsafe_uninit_length=at + out_size)
        Self._decompress_to(method, native, src, Span(dst)[at:])

    def decompress_owned(
        self, native: Bool, src: Span[UInt8, _], out_size: Int
    ) raises -> Buffer[mut=False]:
        """Decompress `src` into a buffer of its own, for a decoder that keeps
        pointing into the page after the reader has moved on."""
        var method = self._check_size(src, out_size)
        var out = Buffer.alloc_uninit[DType.uint8](out_size)
        Self._decompress_to(
            method, native, src, out.view[DType.uint8]().as_span()[:out_size]
        )
        return out^.to_immutable()

    def _check_size(self, src: Span[UInt8, _], out_size: Int) raises -> Method:
        """This codec's method, once `out_size`, from the page header, is
        one `src` can decode to: refused before anything is allocated."""
        var method = self.method()
        if out_size < 0 or out_size > method.max_decompressed_length(len(src)):
            raise CorruptError(
                t"parquet: {len(src)} compressed bytes cannot hold {out_size}"
            )
        return method

    @staticmethod
    def _decompress_to[
        o: Origin[mut=True]
    ](
        method: Method, native: Bool, src: Span[UInt8, _], dst: Span[UInt8, o]
    ) raises:
        method.decompress[
            Method.NONE,
            Method.SNAPPY,
            Method.GZIP,
            Method.BROTLI,
            Method.LZ4_HADOOP,
            Method.ZSTD,
            Method.LZ4,
        ](native, src, dst)

    def compress(
        self, native: Bool, src: Span[mut=False, UInt8, _], mut dst: List[UInt8]
    ) raises:
        """Append `src`, compressed, to `dst`."""
        self.method().compress[
            Method.NONE,
            Method.SNAPPY,
            Method.GZIP,
            Method.BROTLI,
            Method.LZ4_HADOOP,
            Method.ZSTD,
            Method.LZ4,
        ](native, src, dst)
