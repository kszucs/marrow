# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`Method`: a compression method by name, and the one place that runs it
-- marrow's own codec or, with `native` off, the system library's. Parquet's
pages, Arrow IPC's buffers and Avro's blocks each map their codec ids to a
`Method` and keep only their own framing.

Every operation that runs a codec takes the methods its caller supports as
comptime parameters, and only those are compiled in:

```mojo
Method.ZSTD.compress[Method.LZ4_FRAME, Method.ZSTD](True, src, out)
```

A method outside that list raises `NotImplementedError`.
"""

from ..errors import CorruptError, DynError, NotImplementedError
from ..views import BufferView
from .brotli import Brotli, LibBrotli
from .core import Compressor, compress_with, decompress_with
from .hadoop import Hadoop
from .lz4 import LibLz4, LibLz4Frame, Lz4, Lz4Frame
from .snappy import LibSnappy, Snappy
from .zlib import LibDeflate, LibGzip
from .zstd import LibZstd, Zstd


@fieldwise_init
struct Method(Equatable, ImplicitlyCopyable, Movable, Writable):
    """A compression method: what a format's codec id names, before its
    framing."""

    var code: Int

    comptime NONE = Self(0)
    """Stored as is."""
    comptime SNAPPY = Self(1)
    """A raw Snappy block."""
    comptime GZIP = Self(2)
    """Deflate behind a gzip header and trailer, through zlib."""
    comptime DEFLATE = Self(3)
    """Raw RFC 1951 deflate, through zlib."""
    comptime BROTLI = Self(4)
    """A Brotli stream; compressed at quality 1 by marrow's encoder, at
    quality 11 by libbrotli."""
    comptime LZ4 = Self(5)
    """A raw LZ4 block."""
    comptime LZ4_FRAME = Self(6)
    """An LZ4 frame."""
    comptime LZ4_HADOOP = Self(7)
    """LZ4 blocks in Hadoop frames (`Hadoop`)."""
    comptime ZSTD = Self(8)
    """A Zstandard frame."""

    comptime _NAMES: Array[StaticString, 9] = [
        "none",
        "snappy",
        "gzip",
        "deflate",
        "brotli",
        "lz4",
        "lz4_frame",
        "lz4_hadoop",
        "zstd",
    ]

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        return self.code == other.code

    def write_to[W: Writer](self, mut writer: W):
        var names = materialize[Self._NAMES]()
        if 0 <= self.code < len(names):
            writer.write(names[self.code])
        else:
            writer.write("unknown(", self.code, ")")

    def needs_libs(self, native: Bool) -> Bool:
        """Whether decompressing by this method may call into a `dlopen`ed
        library: GZIP and DEFLATE always, NONE never, and the rest unless
        `native`."""
        if self == Self.NONE:
            return False
        elif self == Self.GZIP or self == Self.DEFLATE:
            return True
        else:
            return not native

    def _unsupported(self) -> NotImplementedError:
        return NotImplementedError(t"{self} compression is not supported here")

    def max_decompressed_length(self, n: Int) -> Int:
        """The most `n` bytes compressed by this method decode to."""
        if self == Self.NONE:
            return n
        elif self == Self.SNAPPY:
            return Snappy.max_decompressed_length(n)
        elif self == Self.GZIP or self == Self.DEFLATE:
            return LibGzip.max_decompressed_length(n)
        elif self == Self.BROTLI:
            return Brotli.max_decompressed_length(n)
        elif self == Self.LZ4 or self == Self.LZ4_HADOOP:
            return Lz4.max_decompressed_length(n)
        elif self == Self.LZ4_FRAME:
            return Lz4Frame.max_decompressed_length(n)
        else:
            return Zstd.max_decompressed_length(n)

    def compress[
        *allowed: Method
    ](
        self, native: Bool, src: Span[mut=False, UInt8, _], mut dst: List[UInt8]
    ) raises DynError:
        """Append `src`, compressed, to `dst`."""
        comptime for m in allowed:
            if self == m:
                comptime if m == Method.NONE:
                    dst.extend(src)
                elif m == Method.SNAPPY:
                    compress_with[Snappy, LibSnappy](native, src, dst)
                elif m == Method.GZIP:
                    LibGzip.compress(src, dst)
                elif m == Method.DEFLATE:
                    LibDeflate.compress(src, dst)
                elif m == Method.BROTLI:
                    compress_with[Brotli, LibBrotli](native, src, dst)
                elif m == Method.LZ4:
                    compress_with[Lz4, LibLz4](native, src, dst)
                elif m == Method.LZ4_FRAME:
                    compress_with[Lz4Frame, LibLz4Frame](native, src, dst)
                elif m == Method.LZ4_HADOOP:
                    compress_with[Hadoop[Lz4], Hadoop[LibLz4]](native, src, dst)
                else:
                    compress_with[Zstd, LibZstd](native, src, dst)
                return
        raise self._unsupported()

    def decompress[
        o: Origin[mut=True], //, *allowed: Method
    ](
        self, native: Bool, src: Span[UInt8, _], dst: Span[UInt8, o]
    ) raises DynError:
        """Decompress `src` into `dst`, which is exactly as long as what was
        compressed."""
        comptime for m in allowed:
            if self == m:
                comptime if m == Method.NONE:
                    if len(src) < len(dst):
                        raise CorruptError(
                            t"{len(src)} stored bytes cannot hold {len(dst)}"
                        )
                    BufferView(dst).copy_from(src[: len(dst)])
                elif m == Method.SNAPPY:
                    decompress_with[Snappy, LibSnappy](native, src, dst)
                elif m == Method.GZIP:
                    LibGzip.decompress(src, dst)
                elif m == Method.DEFLATE:
                    LibDeflate.decompress(src, dst)
                elif m == Method.BROTLI:
                    decompress_with[Brotli, LibBrotli](native, src, dst)
                elif m == Method.LZ4:
                    decompress_with[Lz4, LibLz4](native, src, dst)
                elif m == Method.LZ4_FRAME:
                    decompress_with[Lz4Frame, LibLz4Frame](native, src, dst)
                elif m == Method.LZ4_HADOOP:
                    decompress_with[Hadoop[Lz4], Hadoop[LibLz4]](
                        native, src, dst
                    )
                else:
                    decompress_with[Zstd, LibZstd](native, src, dst)
                return
        raise self._unsupported()

    def decompress_unsized[
        *allowed: Method
    ](self, native: Bool, src: Span[UInt8, _]) raises DynError -> List[UInt8]:
        """`src` decompressed, for a format that does not record how long
        that is: SNAPPY by the length the stream declares, ZSTD so too when
        `native` and it declares one -- else by libzstd's streaming decoder
        -- and GZIP and DEFLATE by inflating until the stream ends. The
        other methods need the length; they raise `NotImplementedError`."""
        comptime for m in allowed:
            if self == m:
                comptime if m == Method.NONE:
                    return List[UInt8](src)
                elif m == Method.SNAPPY:
                    return _declared[Snappy, LibSnappy](
                        native, Snappy.content_length(src).value(), src
                    )
                elif m == Method.GZIP:
                    return LibGzip.decompress_unsized(src)
                elif m == Method.DEFLATE:
                    return LibDeflate.decompress_unsized(src)
                elif m == Method.ZSTD:
                    if native:
                        var n = Zstd.content_length(src)
                        if n:
                            return _declared[Zstd, LibZstd](
                                True, n.value(), src
                            )
                    return LibZstd.decompress_unsized(src)
                else:
                    raise NotImplementedError(
                        t"{self} decompression needs the decompressed length"
                    )
        raise self._unsupported()


def _declared[
    Native: Compressor, Library: Compressor
](native: Bool, n: Int, src: Span[UInt8, _]) raises DynError -> List[UInt8]:
    """`src` decompressed into the `n` bytes it declares, as
    `decompress_with` chooses. `n` comes from the native codec's
    `content_length`, which refuses a length `src` cannot hold."""
    var out = List[UInt8](unsafe_uninit_length=n)
    decompress_with[Native, Library](native, src, Span(out))
    return out^
