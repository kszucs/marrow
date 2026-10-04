# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Single-shot profiling driver for `marrow.utils.snappy`.

Runs one operation over one of `bench_snappy.mojo`'s 1 MiB corpora in a loop,
so a sampling profiler sees nothing but that operation. Build it at `-O3`,
the level the benchmarks measure -- the codec is one inlined loop, and an
`-O1` profile would be of different code:

    pixi run -e dev mojo build -O3 -g --debug-info-language C -I . \\
        benchmarks/profiles/profile_snappy.mojo -o /tmp/profile_snappy

`MARROW_PROFILE_OP` picks the operation -- `compress`, `decompress`, `pair`,
or `lib_compress` / `lib_decompress` for libsnappy's, to compare against --
`MARROW_PROFILE_CORPUS` the corpus (`strings`, `ints`, `floats`) and
`MARROW_PROFILE_ITERS` the number of calls (default 4000). On macOS it
prints its image's load address, which maps a sample's PC to a source line
(`atos -i -o BINARY -l LOAD PC`) when the profiler could not unwind the stack.
"""

from std.benchmark import keep
from std.ffi import external_call
from std.os.env import getenv
from std.sys.info import CompilationTarget

from marrow.utils import CompressionLibs, Snappy
from marrow.utils.tests.codec_data import LibSnappy, corpus

comptime N = 1 << 20


def main() raises:
    var op = getenv("MARROW_PROFILE_OP", "decompress")
    var kind = getenv("MARROW_PROFILE_CORPUS", "strings")
    var iters_env = getenv("MARROW_PROFILE_ITERS", "")
    var iters = Int(iters_env) if iters_env.byte_length() > 0 else 4000
    print(t"profile_snappy: {op} {kind} x {iters}")
    comptime if CompilationTarget.is_macos():
        # Instruments drops the backtrace of a sample it cannot unwind -- no
        # frame pointers at -O3 -- but keeps its PC; mapping that to a line
        # needs the image's load address.
        var image = external_call["_dyld_get_image_header", Int](UInt32(0))
        print(t"profile_snappy: image at {hex(image)}")

    var data = corpus(kind, 2 * N)
    var a = List[UInt8](Span(data)[:N])
    var b = List[UInt8](Span(data)[N:])
    var libs = CompressionLibs()
    var ca = libs.snappy_compress(Span(a))
    var cb = libs.snappy_compress(Span(b))
    var oa = List[UInt8](length=N, fill=0)
    var ob = List[UInt8](length=N, fill=0)
    var out = List[UInt8](capacity=Snappy.max_compressed_length(N))
    var lib_out = List[UInt8](length=Snappy.max_compressed_length(N), fill=0)
    var lib = LibSnappy()

    for _ in range(iters):
        if op == "compress":
            out.clear()
            Snappy.compress(Span(a), out)
        elif op == "decompress":
            Snappy.decompress_into(Span(ca), Span(oa))
        elif op == "pair":
            Snappy.decompress_pair_into(Span(ca), Span(oa), Span(cb), Span(ob))
        elif op == "lib_compress":
            keep(lib.compress(Span(a), lib_out))
        elif op == "lib_decompress":
            lib.decompress(Span(ca), Span(oa))
        else:
            raise Error(t"unknown MARROW_PROFILE_OP: {op}")
    keep(out)
    keep(lib_out)
    keep(oa)
    keep(ob)
    print("profile_snappy: done")
