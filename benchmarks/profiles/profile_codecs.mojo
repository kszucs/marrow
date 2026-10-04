# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Single-shot profiling driver for `marrow.utils.zstd` and `.lz4`.

Runs one codec operation over one of the codec benchmarks' corpora in a
loop, so a sampling profiler sees nothing but that:

    pixi run devkit profile --sample benchmarks/profiles/profile_codecs.mojo

`MARROW_PROFILE_OP` picks `decompress`, `compress`, or `lib_decompress` /
`lib_compress` (libzstd's at the same level, to compare against), and the
same four prefixed `lz4_` for the LZ4 block codec against liblz4;
`MARROW_PROFILE_CORPUS` the corpus (`strings`, `ints`, `floats`),
`MARROW_PROFILE_SIZE` its bytes (default 1 MiB),
`MARROW_PROFILE_LEVEL` the level libzstd compresses it at (default 1) and
`MARROW_PROFILE_ITERS` the number of calls (default 2000). On macOS it prints
its image's load address, which maps a sample's PC to a source line
(`atos -i -o BINARY -l LOAD PC`) when the profiler could not unwind the
stack.
"""

from std.benchmark import keep
from std.ffi import external_call
from std.os.env import getenv
from std.sys.info import CompilationTarget

from marrow.utils import Lz4, Zstd
from marrow.utils.tests.codec_data import LibLz4, LibZstd, corpus


def _int(name: String, default: Int) raises -> Int:
    var value = getenv(name, "")
    return Int(value) if value.byte_length() > 0 else default


def main() raises:
    var op = getenv("MARROW_PROFILE_OP", "decompress")
    var kind = getenv("MARROW_PROFILE_CORPUS", "floats")
    var n = _int("MARROW_PROFILE_SIZE", 1 << 20)
    var level = _int("MARROW_PROFILE_LEVEL", 1)
    var iters = _int("MARROW_PROFILE_ITERS", 2000)
    print(t"profile_codecs: {op} {kind} {n} bytes, level {level}, x {iters}")
    comptime if CompilationTarget.is_macos():
        var image = external_call["_dyld_get_image_header", Int](UInt32(0))
        print(t"profile_codecs: image at {hex(image)}")

    var data = corpus(kind, n)
    var lib = LibZstd()
    var frame = lib.compress(Span(data), level)
    var lz4 = LibLz4()
    var block = lz4.compress(Span(data))
    var out = List[UInt8](length=n, fill=0)
    var packed = List[UInt8](capacity=Zstd.max_compressed_length(n))
    for _ in range(iters):
        if op == "decompress":
            Zstd.decompress_into(Span(frame), Span(out))
        elif op == "lib_decompress":
            lib.decompress(Span(frame), Span(out))
        elif op == "compress":
            packed.clear()
            Zstd.compress(Span(data), packed)
        elif op == "lib_compress":
            packed.clear()
            _ = lib.compress_into(Span(data), packed, level)
        elif op == "lz4_decompress":
            Lz4.decompress_block_into(Span(block), Span(out))
        elif op == "lz4_lib_decompress":
            lz4.decompress(Span(block), Span(out))
        elif op == "lz4_compress":
            packed.clear()
            Lz4.compress_block(Span(data), packed)
        elif op == "lz4_lib_compress":
            packed.clear()
            _ = lz4.compress_into(Span(data), packed)
        else:
            raise Error(t"unknown MARROW_PROFILE_OP: {op}")
    keep(out)
    keep(packed)
    if "decompress" in op and out != data:
        raise Error("profile_codecs: decoded wrong")
    print("profile_codecs: done")
