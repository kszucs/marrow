# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

# `Hadoop` is ported from Arrow C++'s `Lz4HadoopCodec`
# (`cpp/src/arrow/util/compression_lz4.cc`), Apache-2.0; see NOTICE.txt.

"""`Hadoop`, the framing Hadoop's codecs write: Parquet's deprecated `LZ4`
codec (`Method.LZ4_HADOOP`)."""

from ..errors import DynError
from ..utils.byteorder import BigEndian, LittleEndian
from .core import Compressor


struct Hadoop[C: Compressor](Compressor):
    """Blocks of `C` in Hadoop frames -- a big-endian u32 decompressed size,
    a big-endian u32 compressed size, then the block -- as Arrow C++ writes
    Parquet's deprecated `LZ4` codec, one frame a page. Decoding takes a run
    of frames or, when the bytes do not parse as one, a plain block: what
    Parquet C++ wrote before it framed them."""

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
