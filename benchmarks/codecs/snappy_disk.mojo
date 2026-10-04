# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Snappy page decompression from disk, not just from memory.

A column chunk of 8 independently snappy-compressed 1 MiB pages is written
to a file and read the way marrow reads a local Parquet file: `BufferSource`
memory-maps it and the decoder reads the compressed bytes straight from the
mapping, so a page is faulted in from disk the first time the decoder touches
it. There is no separate read step to time.

**Cold** runs decode a copy written moments earlier and kept out of the page
cache -- `F_NOCACHE` on macOS, `POSIX_FADV_DONTNEED` once it is synced on
Linux -- so every fault goes to the disk.
**Warm** runs decode a file already in the page cache -- what
`bench_snappy.mojo` measures, plus the mapping's soft faults. `touch` reads
one byte per 4 KiB of the mapping and decodes nothing: the disk alone.
`+willneed` variants call `madvise(MADV_WILLNEED)` on the mapping first, so
the kernel reads the whole chunk ahead instead of fault by fault.

Each trial runs every variant once, interleaved, and the best of the trials is
reported -- the machine is shared, so the minimum is the undisturbed run. Wall
clock, because waiting on the disk is the point.

Build and run from the repository root:

    pixi run -e dev mojo build -O3 -g1 -I . benchmarks/codecs/snappy_disk.mojo -o /tmp/snappy_disk
    /tmp/snappy_disk

`MARROW_SNAPPY_TRIALS` overrides the trial count (default 10).
"""

from std.benchmark import keep
from std.ffi import external_call
from std.os import remove
from std.os.env import getenv
from std.os.path import join
from std.python import Python
from std.time import perf_counter_ns

from marrow.io.local import BufferSource
from marrow.utils import CompressionLibs, Snappy
from marrow.utils.testing import ScratchDir
from marrow.utils.tests.codec_data import LibSnappy, corpus

comptime PAGES = 8
comptime PAGE = 1 << 20

comptime _COLD_COPY = """
import os, sys, fcntl

def cold_copy(src, dst):
    darwin = sys.platform == "darwin"
    if not darwin and not hasattr(os, "posix_fadvise"):
        raise OSError(f"cannot keep a file out of the page cache on {sys.platform}")
    with open(src, "rb") as f:
        data = f.read()
    fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        if darwin:
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
        if not darwin:
            # Linux caches what it writes; once synced, the pages are clean
            # and can be dropped.
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
"""


struct Chunk:
    """Where each compressed page sits in the file."""

    var start: List[Int]
    var length: List[Int]

    def __init__(out self):
        self.start = List[Int]()
        self.length = List[Int]()

    def size(self) -> Int:
        """The file's length: the compressed pages back to back."""
        return self.start[PAGES - 1] + self.length[PAGES - 1]


def _write_chunk(data: List[UInt8], path: String) raises -> Chunk:
    var libs = CompressionLibs()
    var chunk = Chunk()
    var body = List[UInt8]()
    for p in range(PAGES):
        var comp = libs.snappy_compress(Span(data)[p * PAGE : (p + 1) * PAGE])
        chunk.start.append(len(body))
        chunk.length.append(len(comp))
        body.extend(Span(comp))
    with open(path, "w") as f:
        f.write_bytes(Span(body))
    return chunk^


def _run(
    variant: String,
    path: String,
    chunk: Chunk,
    mut snappy: LibSnappy,
    mut out: List[UInt8],
    mut odd: List[UInt8],
) raises -> Int:
    """Map `path`, do `variant`'s work over it; nanoseconds, map included."""
    var t0 = perf_counter_ns()
    var src = BufferSource(path)
    var size = chunk.size()
    if variant.endswith("+willneed"):
        # Ask the kernel to start reading the whole chunk now, so the decoder
        # faults on pages already in flight rather than one region at a time.
        _ = external_call["madvise", Int32](
            src.read_at(0, size).unsafe_ptr(), size, Int32(3)  # MADV_WILLNEED
        )
    var base = variant.removesuffix("+willneed")
    if base == "touch":
        var acc = UInt64(0)
        var all = src.read_at(0, size)
        for i in range(0, size, 4096):
            acc += UInt64(all[i])
        keep(acc)
    elif base == "libsnappy":
        for p in range(PAGES):
            snappy.decompress(
                src.read_at(chunk.start[p], chunk.length[p]),
                Span(out)[p * PAGE : (p + 1) * PAGE],
            )
    elif base == "native":
        for p in range(PAGES):
            Snappy.decompress_into(
                src.read_at(chunk.start[p], chunk.length[p]),
                Span(out)[p * PAGE : (p + 1) * PAGE],
            )
    else:
        # Two destinations, as a reader's two page scratch buffers would be:
        # even pages into `out`, odd pages packed into `odd`.
        for p in range(0, PAGES, 2):
            var o = (p // 2) * PAGE
            Snappy.decompress_pair_into(
                src.read_at(chunk.start[p], chunk.length[p]),
                Span(out)[p * PAGE : (p + 1) * PAGE],
                src.read_at(chunk.start[p + 1], chunk.length[p + 1]),
                Span(odd)[o : o + PAGE],
            )
    var t1 = perf_counter_ns()
    keep(out[len(out) - 1])
    _ = src^
    return Int(t1 - t0)


def main() raises:
    var trials_env = getenv("MARROW_SNAPPY_TRIALS", "")
    var trials = Int(trials_env) if trials_env.byte_length() > 0 else 10
    var g = Python.dict()
    _ = Python.import_module("builtins").exec(_COLD_COPY, g)
    var cold_copy = g["cold_copy"]
    var variants: List[String] = [
        "touch",
        "libsnappy",
        "native",
        "native_pair",
        "libsnappy+willneed",
        "native_pair+willneed",
    ]
    comptime LIB = 1  # `variants[LIB]`, what every ratio is against
    var snappy = LibSnappy()
    var out = List[UInt8](length=PAGES * PAGE, fill=0)
    var odd = List[UInt8](length=PAGES // 2 * PAGE, fill=0)
    print(
        "corpus   variant       compressed MB   cold ms   warm ms   cold GB/s"
        "   warm GB/s   cold vs libsnappy   warm vs libsnappy"
    )
    with ScratchDir() as dir:
        for kind in ["strings", "ints", "floats"]:
            var master = join(dir, String(kind) + ".chunk")
            var data = corpus(String(kind), PAGES * PAGE)
            var chunk = _write_chunk(data, master)
            var mb = Float64(chunk.size()) / 1e6
            var cold = List[Int](length=len(variants), fill=Int.MAX)
            var warm = List[Int](length=len(variants), fill=Int.MAX)
            # One warm-up pass so the master file sits in the page cache.
            _ = _run("touch", master, chunk, snappy, out, odd)
            for t in range(trials):
                for v in range(len(variants)):
                    var copy = join(dir, String(t"cold_{t}_{v}.chunk"))
                    _ = cold_copy(master, copy)
                    cold[v] = min(
                        cold[v],
                        _run(variants[v], copy, chunk, snappy, out, odd),
                    )
                    remove(copy)
                    warm[v] = min(
                        warm[v],
                        _run(variants[v], master, chunk, snappy, out, odd),
                    )
            # The last run was a pair decode: even pages in `out`, odd in `odd`.
            for p in range(0, PAGES, 2):
                var o = (p // 2) * PAGE
                for i in range(PAGE):
                    if out[p * PAGE + i] != data[p * PAGE + i]:
                        raise Error(t"{kind}: page {p} wrong at byte {i}")
                    if odd[o + i] != data[(p + 1) * PAGE + i]:
                        raise Error(t"{kind}: page {p + 1} wrong at byte {i}")
            var gb = Float64(PAGES * PAGE)
            for v in range(len(variants)):
                var vs_cold = Float64(cold[LIB]) / Float64(cold[v])
                var vs_warm = Float64(warm[LIB]) / Float64(warm[v])
                print(
                    t"{kind}  {variants[v]}  {mb}  {Float64(cold[v]) / 1e6}"
                    t"  {Float64(warm[v]) / 1e6}  {gb / Float64(cold[v])}"
                    t"  {gb / Float64(warm[v])}  {vs_cold}  {vs_warm}"
                )
