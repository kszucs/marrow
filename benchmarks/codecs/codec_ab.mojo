# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The native LZ4 and Zstandard codecs against liblz4 and libzstd, measured
so that a busy machine cannot decide the answer.

Wall-clock benchmarks on a shared machine swing 2-5x with the load. This
alternates the two sides batch by batch in one process and keeps each side's
best *thread CPU time* over many rounds, so both see the same clock frequency
and neither is charged for time it spent descheduled:

    pixi run -e dev mojo build -O3 -I . benchmarks/codecs/codec_ab.mojo \\
        -o /tmp/codec_ab && /tmp/codec_ab

`MARROW_AB_ROUNDS` sets the rounds (default 15) and `MARROW_AB_FILTER` keeps
only the rows whose name contains it. A ratio under 1.00 is the native codec
faster.
"""

from std.benchmark import keep
from std.ffi import external_call
from std.os.env import getenv
from std.sys.info import CompilationTarget

from marrow.utils import Lz4, Zstd
from marrow.utils.tests.codec_data import LibLz4, LibZstd, corpus


def _cpu_ns() -> Int:
    """This thread's CPU time."""
    comptime if CompilationTarget.is_macos():
        # CLOCK_THREAD_CPUTIME_ID is 16 on Darwin.
        return Int(external_call["clock_gettime_nsec_np", UInt64](UInt32(16)))
    else:
        var ts = Array[Int64, 2](fill=0)
        # CLOCK_THREAD_CPUTIME_ID is 3 on Linux.
        _ = external_call["clock_gettime", Int32](Int32(3), ts.unsafe_ptr())
        return Int(ts[0]) * 1_000_000_000 + Int(ts[1])


struct _Case(Movable):
    var data: List[UInt8]
    var lz4_block: List[UInt8]
    var zstd_frame: List[UInt8]
    var out: List[UInt8]
    var packed: List[UInt8]
    """Either side's compressed output, appended within a capacity reserved
    once, so neither side's timed call grows it."""

    def __init__(out self, kind: String, n: Int) raises:
        self.data = corpus(kind, n)
        self.lz4_block = LibLz4().compress(Span(self.data))
        self.zstd_frame = LibZstd().compress(Span(self.data), 1)
        self.out = List[UInt8](length=n, fill=0)
        self.packed = List[UInt8](capacity=Zstd.max_compressed_length(n) + n)


def _run(
    op: String, native: Bool, mut c: _Case, lz4: LibLz4, zstd: LibZstd
) raises:
    """One call of `op` on one side."""
    if op == "lz4 compress":
        c.packed.clear()
        if native:
            Lz4.compress_block(Span(c.data), c.packed)
        else:
            keep(lz4.compress_into(Span(c.data), c.packed))
    elif op == "lz4 decompress":
        if native:
            Lz4.decompress_block_into(Span(c.lz4_block), Span(c.out))
        else:
            lz4.decompress(Span(c.lz4_block), Span(c.out))
    elif op == "zstd compress":
        c.packed.clear()
        if native:
            Zstd.compress(Span(c.data), c.packed)
        else:
            keep(zstd.compress_into(Span(c.data), c.packed, 1))
    else:
        if native:
            Zstd.decompress_into(Span(c.zstd_frame), Span(c.out))
        else:
            zstd.decompress(Span(c.zstd_frame), Span(c.out))
    keep(c.out)
    keep(c.packed)


def _pad(text: String, width: Int, left: Bool = False) -> String:
    var fill = String(" ") * max(0, width - text.byte_length())
    return text + fill if left else fill + text


def main() raises:
    var rounds_env = getenv("MARROW_AB_ROUNDS", "")
    var rounds = Int(rounds_env) if rounds_env.byte_length() > 0 else 15
    var only = getenv("MARROW_AB_FILTER", "")
    var lz4 = LibLz4()
    var zstd = LibZstd()
    var ops: List[String] = [
        "lz4 compress",
        "lz4 decompress",
        "zstd compress",
        "zstd decompress",
    ]
    print("op                 corpus      size   lib us  native us  native/lib")
    for op in ops:
        for kind in ["strings", "ints", "floats"]:
            for n in [1 << 16, 1 << 20]:
                var name = String(op, " ", kind, " ", n)
                if only.byte_length() > 0 and only not in name:
                    continue
                var c = _Case(kind, n)
                # A batch is about 4 MiB of input, whatever the size.
                var batch = max(1, (4 << 20) // n)
                var best = [Int.MAX, Int.MAX]
                for _ in range(rounds):
                    for side in range(2):
                        var t0 = _cpu_ns()
                        for _ in range(batch):
                            _run(op, side == 1, c, lz4, zstd)
                        best[side] = min(best[side], _cpu_ns() - t0)
                var lib_us = Float64(best[0]) / Float64(batch) / 1000
                var native_us = Float64(best[1]) / Float64(batch) / 1000
                print(
                    _pad(op, 18, left=True),
                    _pad(kind, 8, left=True),
                    _pad(String(n), 8),
                    _pad(String(Int(lib_us)), 8),
                    _pad(String(Int(native_us)), 10),
                    _pad(String(round(native_us / lib_us, 2)), 11),
                )
