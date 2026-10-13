# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Deflate (RFC 1951) through the system's zlib: `LibGzip`, Parquet's GZIP,
and `LibDeflate`, Avro's `deflate`. Deflate has no codec in Mojo, so unlike
the other formats' modules this one holds only the library's.

Avro's blocks record no uncompressed size, which is what
`LibDeflate.decompress_unsized` is for: it grows its output rather than fill
an exact one.
"""

from std.memory import unsafe_memset_zero
from std.memory.alloc import unsafe_alloc

from ..errors import CorruptError, DynError, InternalError
from .core import Compressor, append_uninit, library


struct _ZStream(Movable):
    """A zlib `z_stream`, driven directly: `SIZE` bytes on LP64, of which
    marrow touches next_in @0, avail_in @8, next_out @24, avail_out @32 and
    total_out @40. Zeroed, so zalloc/zfree/opaque select zlib's defaults."""

    comptime SIZE = 112
    """`sizeof(z_stream)`, which the init calls check against their own."""

    var _p: Pointer[UInt64, MutUntrackedOrigin]

    def __init__(out self):
        self._p = unsafe_alloc[UInt64](16)
        unsafe_memset_zero(self._p.unsafe_bitcast[UInt8](), 128)

    def __deinit__(deinit self):
        self._p.unsafe_free()

    def ptr(self) -> Pointer[UInt8, MutUntrackedOrigin]:
        return self._p.unsafe_bitcast[UInt8]()

    def set_input(self, src: Pointer[UInt8, _], n: Int):
        self._p[unsafe_offset=0] = UInt64(Int(src))
        self.ptr().unsafe_offset(8).unsafe_bitcast[UInt32]()[
            unsafe_offset=0
        ] = UInt32(n)

    def set_output(self, dst: Pointer[UInt8, _], n: Int):
        self._p[unsafe_offset=3] = UInt64(Int(dst))
        self.ptr().unsafe_offset(32).unsafe_bitcast[UInt32]()[
            unsafe_offset=0
        ] = UInt32(n)

    def avail_in(self) -> Int:
        return Int(
            self.ptr()
            .unsafe_offset(8)
            .unsafe_bitcast[UInt32]()[unsafe_offset=0]
        )

    def avail_out(self) -> Int:
        return Int(
            self.ptr()
            .unsafe_offset(32)
            .unsafe_bitcast[UInt32]()[unsafe_offset=0]
        )

    def total_out(self) -> Int:
        return Int(self._p[unsafe_offset=5])


comptime _Z_OK = 0
comptime _Z_STREAM_END = 1
comptime _Z_BUF_ERROR = -5
comptime _Z_FINISH = 4
comptime _GZIP_WINDOW = 31
"""windowBits 15 plus 16: a gzip header and trailer."""
comptime _RAW_WINDOW = -15
"""windowBits -15: raw RFC 1951 deflate, no header or checksum."""


struct LibZlib[window: Int](Compressor):
    """Deflate through zlib, at level 6, framed by `window`: `LibGzip`, Parquet's
    GZIP, or `LibDeflate`, raw RFC 1951 -- Avro's `deflate`."""

    @staticmethod
    def max_decompressed_length(n: Int) -> Int:
        """1032 times as many: deflate's densest, a run of 258-byte copies
        in under two bits each."""
        return 1032 * n

    @staticmethod
    def compress(
        src: Span[mut=False, UInt8, _], mut dst: List[UInt8]
    ) raises DynError:
        var z = library["z"]()
        var strm = _ZStream()
        var version = z.call[
            "zlibVersion", Pointer[UInt8, MutUntrackedOrigin]
        ]()
        # level 6, method Z_DEFLATED(8), memLevel 8, strategy
        # Z_DEFAULT_STRATEGY(0).
        var rc = z.call["deflateInit2_", Int32](
            strm.ptr(),
            Int32(6),
            Int32(8),
            Int32(Self.window),
            Int32(8),
            Int32(0),
            version,
            Int32(_ZStream.SIZE),
        )
        if Int(rc) != _Z_OK:
            raise InternalError("zlib: deflateInit2 failed")
        var bound = Int(
            z.call["deflateBound", UInt](strm.ptr(), UInt(len(src)))
        )
        var at = append_uninit(dst, bound)
        strm.set_input(src.unsafe_ptr(), len(src))
        strm.set_output(dst.unsafe_ptr().unsafe_offset(at), bound)
        var st = z.call["deflate", Int32](strm.ptr(), Int32(_Z_FINISH))
        var produced = strm.total_out()
        _ = z.call["deflateEnd", Int32](strm.ptr())
        if Int(st) != _Z_STREAM_END:
            dst.shrink(at)
            raise InternalError("zlib: deflate failed")
        dst.shrink(at + produced)

    @staticmethod
    def decompress[
        o: Origin[mut=True]
    ](src: Span[UInt8, _], dst: Span[UInt8, o]) raises DynError:
        var z = library["z"]()
        var strm = _ZStream()
        strm.set_input(src.unsafe_ptr(), len(src))
        strm.set_output(dst.unsafe_ptr(), len(dst))
        Self._inflate_init(strm)
        var st = z.call["inflate", Int32](strm.ptr(), Int32(_Z_FINISH))
        var produced = strm.total_out()
        _ = z.call["inflateEnd", Int32](strm.ptr())
        if Int(st) != _Z_STREAM_END or produced != len(dst):
            raise CorruptError("zlib: could not inflate the expected size")

    @staticmethod
    def decompress_unsized(src: Span[UInt8, _]) raises DynError -> List[UInt8]:
        """Inflate a stream whose decompressed size is not known up front --
        Avro's `deflate` blocks record none -- growing the output until the
        stream ends."""
        var z = library["z"]()
        var strm = _ZStream()
        strm.set_input(src.unsafe_ptr(), len(src))
        Self._inflate_init(strm)
        var out = List[UInt8](unsafe_uninit_length=max(64, 4 * len(src)))
        var failure = String()
        while True:
            var done = strm.total_out()
            strm.set_output(
                out.unsafe_ptr().unsafe_offset(done), len(out) - done
            )
            var st = Int(z.call["inflate", Int32](strm.ptr(), Int32(0)))
            if st == _Z_STREAM_END:
                break
            if st != _Z_OK and st != _Z_BUF_ERROR:
                failure = String(t"zlib: inflate failed ({st})")
                break
            if strm.avail_out() == 0:
                out.resize(unsafe_uninit_length=2 * len(out))
            elif strm.avail_in() == 0:
                failure = "zlib: truncated stream"
                break
        _ = z.call["inflateEnd", Int32](strm.ptr())
        if failure:
            raise CorruptError(failure)
        out.resize(unsafe_uninit_length=strm.total_out())
        return out^

    @staticmethod
    def _inflate_init(strm: _ZStream) raises DynError:
        var z = library["z"]()
        var version = z.call[
            "zlibVersion", Pointer[UInt8, MutUntrackedOrigin]
        ]()
        var rc = z.call["inflateInit2_", Int32](
            strm.ptr(), Int32(Self.window), version, Int32(_ZStream.SIZE)
        )
        if Int(rc) != _Z_OK:
            raise InternalError("zlib: inflateInit2 failed")


comptime LibGzip = LibZlib[_GZIP_WINDOW]
"""Deflate behind a gzip header and trailer: Parquet's GZIP."""

comptime LibDeflate = LibZlib[_RAW_WINDOW]
"""Raw RFC 1951 deflate: Avro's `deflate`."""
