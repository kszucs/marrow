# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""What every compressor shares: the `Compressor` trait, the choice between
marrow's codec and the system library's at run time, the system libraries
themselves, and the growing of an output list.

`library`, `append_uninit` and `reserve_more` are public so the codec
modules can call them, but `marrow.compress` does not re-export them: they
are the package's shared implementation, not an API.
"""

from std.ffi import _DLHandle

from ..errors import CorruptError, DynError
from ..utils.dylib import LibSet, LibSpec


trait Compressor:
    """Bytes to fewer bytes and back, the bytes alone. The size of what
    was compressed is the caller's to keep, unless the format records it:
    `content_length`."""

    @staticmethod
    def content_length(src: Span[UInt8, _]) raises DynError -> Optional[Int]:
        """The length `src` declares it decompresses to, or None when the
        format or this stream leaves it out. A declaration `src` cannot
        hold is refused rather than reported, so a reader may allocate
        what it is told."""
        return None

    @staticmethod
    def compress(
        src: Span[mut=False, UInt8, _], mut dst: List[UInt8]
    ) raises DynError:
        """Append `src`, compressed, to `dst`."""
        ...

    @staticmethod
    def decompress[
        o: Origin[mut=True]
    ](src: Span[UInt8, _], dst: Span[UInt8, o]) raises DynError:
        """Decompress `src` into `dst`, which is exactly as long as what
        was compressed; refuse anything else."""
        ...


def compress_with[
    Native: Compressor, Library: Compressor
](
    native: Bool, src: Span[mut=False, UInt8, _], mut dst: List[UInt8]
) raises DynError:
    """Append `src` compressed to `dst`: by `Native`, marrow's own codec,
    when `native`, or else by `Library`, the system library's."""
    if native:
        Native.compress(src, dst)
    else:
        Library.compress(src, dst)


def decompress_with[
    Native: Compressor, Library: Compressor, o: Origin[mut=True]
](native: Bool, src: Span[UInt8, _], dst: Span[UInt8, o]) raises DynError:
    """Decompress `src` into exactly `dst`, as `compress_with` chooses."""
    if native:
        Native.decompress(src, dst)
    else:
        Library.decompress(src, dst)


trait UnsizedDecompressor(Compressor):
    """A compressor that decompresses a stream of unknown length, growing
    its output as it goes."""

    @staticmethod
    def decompress_unsized(src: Span[UInt8, _]) raises DynError -> List[UInt8]:
        """`src`, decompressed, however long that turns out."""
        ...


def decompress_unsized_with[
    Native: Compressor, Library: UnsizedDecompressor
](native: Bool, src: Span[UInt8, _]) raises DynError -> List[UInt8]:
    """`src` decompressed to the length it turns out to have: by `Native`
    into exactly the length the stream declares, when `native` and it
    declares one, or else by `Library`, streaming."""
    if native:
        var n = Native.content_length(src)
        if n:
            if n.value() < 0:
                raise CorruptError(t"declared length {n.value()} is negative")
            var out = List[UInt8](unsafe_uninit_length=n.value())
            Native.decompress(src, Span(out))
            return out^
    return Library.decompress_unsized(src)


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
"""The system compression libraries every `Lib*` compressor calls into:
what each is called, where to look for it, and one process-global holding
all six. A program that reads only uncompressed Parquet opens nothing,
because nothing touches this until a library method runs. Several threads
about to call into these first open them with `Codecs.preload()`, as
opening races."""
# One global for the set, not one per library: six cost `query_cli` ~16 KB
# of `__text`, and the AOT lane is size-gated.


def library[name: StaticString]() raises DynError -> _DLHandle:
    """The library `name` of `Codecs`, or why it could not be opened."""
    try:
        return Codecs.handle[name]()
    except e:
        raise DynError(e)


def reserve_more(mut dst: List[UInt8], n: Int):
    """Make room for `n` more bytes after `dst`'s, at least doubling its
    capacity when it has to grow, so that outputs appended one after another
    to one list -- an IPC body's buffers -- do not copy it once each."""
    var need = len(dst) + n
    if dst.capacity() < need:
        dst.reserve(max(need, 2 * dst.capacity()))


def append_uninit(mut dst: List[UInt8], n: Int) -> Int:
    """Grow `dst` by `n` bytes left to the caller to write; return where
    they start."""
    var at = len(dst)
    reserve_more(dst, n)
    dst.resize(unsafe_uninit_length=at + n)
    return at
