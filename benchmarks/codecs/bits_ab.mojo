# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Bitmap loads and stores and `BitPack`'s kernels, timed so that a busy
machine cannot decide the answer.

Each row is its own function, so one row's code cannot move another's,
and the best *thread CPU time* over many rounds, so time spent
descheduled is not charged. It reads only public API that every tree has, so
the same file builds against two trees -- before and after a change -- and the
two binaries run alternately, each row compared on its best:

    pixi run -e dev mojo build -O3 -I . benchmarks/codecs/bits_ab.mojo \\
        -o /tmp/bits_ab_after && /tmp/bits_ab_after

`MARROW_AB_ROUNDS` sets the rounds (default 15). Each line is a row name and
its best nanoseconds per element.
"""

from std.benchmark import keep
from std.ffi import external_call
from std.os.env import getenv
from std.sys.info import CompilationTarget

from marrow.buffers import Bitmap
from marrow.codecs import BitPack

comptime N = 1 << 20


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


def _random(width: Int, n: Int) -> List[UInt64]:
    var mask = UInt64(0) if width == 0 else UInt64.MAX >> UInt64(64 - width)
    var x = UInt64(0x9E3779B97F4A7C15)
    var out = List[UInt64](capacity=n)
    for _ in range(n):
        x ^= x << 13
        x ^= x >> 7
        x ^= x << 17
        out.append(x & mask)
    return out^


def _report(name: String, best: Int, n: Int):
    print(name, Float64(best) / Float64(n))


@no_inline
def _load[W: Int](bits: List[UInt64], rounds: Int):
    var bm = Bitmap.alloc_zeroed(N)
    for i in range(N):
        if bits[i] == 1:
            bm.set(i)
    var view = bm.view(3, N - 3)
    var best = Int.MAX
    for _ in range(rounds):
        var t0 = _cpu_ns()
        var acc = SIMD[DType.bool, W](fill=False)
        for i in range(0, N - 3 - W, W):
            acc ^= view.load[W](i)
        keep(acc)
        best = min(best, _cpu_ns() - t0)
    _report(String("bitmap_load_w", W), best, N)


@no_inline
def _store[W: Int](rounds: Int):
    var bm = Bitmap.alloc_zeroed(N)
    var view = bm.view()
    var pattern = SIMD[DType.bool, W](fill=False)
    for i in range(0, W, 3):
        pattern[i] = True
    var best = Int.MAX
    for _ in range(rounds):
        var t0 = _cpu_ns()
        for i in range(0, N - W + 1, W):
            view.store[W](i, pattern)
        keep(view.load_bytes[DType.uint8](0))
        best = min(best, _cpu_ns() - t0)
    _report(String("bitmap_store_w", W), best, N)


@no_inline
def _pack(width: Int, rounds: Int):
    var values = _random(width, N)
    var out = List[UInt8](capacity=(N * width + 7) // 8 + 8)
    var best = Int.MAX
    for _ in range(rounds):
        out.clear()
        var t0 = _cpu_ns()
        BitPack.pack(Span(values), width, out)
        best = min(best, _cpu_ns() - t0)
        keep(len(out))
    _report(String("bitpack_pack_w", width), best, N)


@no_inline
def _unpack(width: Int, rounds: Int):
    var values = _random(width, N)
    var data = List[UInt8]()
    BitPack.pack(Span(values), width, data)
    var out = List[UInt64](capacity=N)
    var best = Int.MAX
    for _ in range(rounds):
        out.clear()
        var t0 = _cpu_ns()
        BitPack.unpack(Span(data), 0, width, N, out)
        best = min(best, _cpu_ns() - t0)
        keep(len(out))
    _report(String("bitpack_unpack_w", width), best, N)


@no_inline
def _get(width: Int, rounds: Int):
    var data = List[UInt8]()
    BitPack.pack(Span(_random(width, N)), width, data)
    var best = Int.MAX
    for _ in range(rounds):
        var t0 = _cpu_ns()
        var acc = UInt64(0)
        for i in range(N):
            acc ^= BitPack.get(Span(data), i * width, width)
        keep(acc)
        best = min(best, _cpu_ns() - t0)
    _report(String("bitpack_get_w", width), best, N)


def main() raises:
    var rounds_env = getenv("MARROW_AB_ROUNDS", "")
    var rounds = Int(rounds_env) if rounds_env.byte_length() > 0 else 15
    var bits = _random(1, N)
    _load[8](bits, rounds)
    _load[16](bits, rounds)
    _load[32](bits, rounds)
    _load[64](bits, rounds)
    _store[8](rounds)
    _store[32](rounds)
    _store[64](rounds)
    for w in [1, 3, 7, 8, 17, 32, 64]:
        _pack(w, rounds)
        _unpack(w, rounds)
    _get(1, rounds)
    _get(17, rounds)
