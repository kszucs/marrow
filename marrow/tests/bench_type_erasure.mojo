"""Benchmark: three type-erasure strategies for immutable array handles.

Measures for each approach:
  A. Erase cost    — typed PrimitiveArray[int64] → erased handle
  B. Dispatch cost — call a method through the erased handle (tight loop)
  C. Copy cost     — copy the erased handle (OwnedHandle is move-only: N/A)

Approaches:
  copy  (AnyArray)  — 7-field struct copy; Buffer/Bitmap are Arc-backed so O(1)
  arc  + vtable     — heap-alloc into ArcPointer + function-pointer trampolines
  owned+ vtable     — heap-alloc into ArcPointer, move-only (no ref-count on copy)

Run with:
  pixi run mojo run -I . marrow/tests/bench_type_erasure.mojo
"""

from std.time import perf_counter_ns
from std.benchmark import keep
from std.memory import ArcPointer, OwnedPointer

from marrow.arrays import AnyArray, PrimitiveArray
from marrow.builders import PrimitiveBuilder
from marrow.dtypes import int64


comptime Arr = PrimitiveArray[int64]
comptime ITERS = 1_000_000
comptime ERASE_ITERS = 100_000


def _make_array(n: Int) raises -> Arr:
    var b = PrimitiveBuilder[int64](n)
    for i in range(n):
        b.unsafe_append(Scalar[int64.native](i))
    return b.finish_typed()


def _ns(elapsed: UInt, iters: Int) -> Float64:
    return Float64(elapsed) / Float64(iters)


# ---------------------------------------------------------------------------
# ArcHandle — vtable + ArcPointer  (mirrors AnyBuilder, copyable)
# ---------------------------------------------------------------------------


struct ArcHandle(Copyable, Movable):
    """Type-erased handle: heap-allocated behind ArcPointer + fn-ptr vtable.

    Copy = 1 atomic ref-count bump + copy of fn-ptr fields.
    Dispatch = fn-ptr call + ArcPointer rebind + field access.
    """

    var _data: ArcPointer[NoneType]
    var _virt_length: def(ArcPointer[NoneType]) -> Int

    @staticmethod
    def _tramp_length(ptr: ArcPointer[NoneType]) -> Int:
        return rebind[ArcPointer[Arr]](ptr)[].length

    def __init__(out self, var arr: Arr):
        var typed = ArcPointer(arr^)
        self._data = rebind[ArcPointer[NoneType]](typed^)
        self._virt_length = Self._tramp_length

    def __init__(out self, *, copy: Self):
        self._data = copy._data
        self._virt_length = copy._virt_length

    def length(self) -> Int:
        return self._virt_length(self._data)


# ---------------------------------------------------------------------------
# OwnedHandle — vtable + OwnedPointer (no atomic refcount, move-only)
# ---------------------------------------------------------------------------


struct OwnedHandle(Movable):
    """Type-erased handle: heap-allocated via OwnedPointer (no refcount).

    Move-only. Erase cost avoids the atomic write that ArcPointer pays on init.
    Dispatch goes through the same fn-ptr indirection as ArcHandle.
    """

    var _owned: OwnedPointer[Arr]
    var _ptr: UnsafePointer[
        Arr, MutAnyOrigin
    ]  # cached at init, avoids origin mismatch
    var _virt_length: def(UnsafePointer[Arr, MutAnyOrigin]) -> Int

    @staticmethod
    def _tramp_length(ptr: UnsafePointer[Arr, MutAnyOrigin]) -> Int:
        return ptr[].length

    def __init__(out self, var arr: Arr):
        self._owned = OwnedPointer(arr^)
        self._ptr = self._owned.unsafe_ptr()
        self._virt_length = Self._tramp_length

    def length(self) -> Int:
        return self._virt_length(self._ptr)


# ---------------------------------------------------------------------------
# A. Erase cost
# ---------------------------------------------------------------------------


def bench_a_erase(arr: Arr) raises:
    """Convert typed → erased handle N times; measures construction overhead."""

    # copy (AnyArray) — struct copy + List[Buffer] alloc + Arc bumps for Buffer
    for _ in range(3):
        var h = arr.as_any()
        keep(h.length)
    var t0 = perf_counter_ns()
    for _ in range(ERASE_ITERS):
        var h = arr.as_any()
        keep(h.length)
    var copy_ns = _ns(perf_counter_ns() - t0, ERASE_ITERS)

    # arc + vtable — heap alloc for ArcPointer + Arc bumps + fn-ptr copy
    for _ in range(3):
        var h = ArcHandle(arr.copy())
        keep(h.length())
    t0 = perf_counter_ns()
    for _ in range(ERASE_ITERS):
        var h = ArcHandle(arr.copy())
        keep(h.length())
    var arc_ns = _ns(perf_counter_ns() - t0, ERASE_ITERS)

    # owned + vtable — same heap alloc, move-only (no extra atomic on init)
    for _ in range(3):
        var h = OwnedHandle(arr.copy())
        keep(h.length())
    t0 = perf_counter_ns()
    for _ in range(ERASE_ITERS):
        var h = OwnedHandle(arr.copy())
        keep(h.length())
    var owned_ns = _ns(perf_counter_ns() - t0, ERASE_ITERS)

    print("A. Erase cost (ns/op,", ERASE_ITERS, "iters):")
    print("   copy  (AnyArray) :", copy_ns)
    print("   arc  + vtable    :", arc_ns)
    print("   owned+ vtable    :", owned_ns)


# ---------------------------------------------------------------------------
# B. Dispatch cost
# ---------------------------------------------------------------------------


def bench_b_dispatch(arr: Arr) raises:
    """Pre-create one handle, call length() through it M times."""
    var copy_h = arr.as_any()
    var arc_h = ArcHandle(arr.copy())
    var owned_h = OwnedHandle(arr.copy())

    # copy (AnyArray) — direct struct field access, no indirection
    for _ in range(3):
        keep(copy_h.length)
    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        keep(copy_h.length)
    var copy_ns = _ns(perf_counter_ns() - t0, ITERS)

    # arc + vtable — fn-ptr call + ArcPointer rebind + field access
    for _ in range(3):
        keep(arc_h.length())
    t0 = perf_counter_ns()
    for _ in range(ITERS):
        keep(arc_h.length())
    var arc_ns = _ns(perf_counter_ns() - t0, ITERS)

    # owned + vtable — identical dispatch path to arc
    for _ in range(3):
        keep(owned_h.length())
    t0 = perf_counter_ns()
    for _ in range(ITERS):
        keep(owned_h.length())
    var owned_ns = _ns(perf_counter_ns() - t0, ITERS)

    print("B. Dispatch cost (ns/op,", ITERS, "iters):")
    print("   copy  (AnyArray) :", copy_ns)
    print("   arc  + vtable    :", arc_ns)
    print("   owned+ vtable    :", owned_ns)


# ---------------------------------------------------------------------------
# C. Copy cost
# ---------------------------------------------------------------------------


def bench_c_copy(arr: Arr) raises:
    """Copy the erased handle K times; OwnedHandle is move-only (skipped)."""
    var copy_h = arr.as_any()
    var arc_h = ArcHandle(arr.copy())

    # copy (AnyArray) — struct copy: 7 fields + Arc bumps + List copies
    for _ in range(3):
        var h2 = copy_h.copy()
        keep(h2.length)
    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        var h2 = copy_h.copy()
        keep(h2.length)
    var copy_ns = _ns(perf_counter_ns() - t0, ITERS)

    # arc + vtable — 1 atomic bump + copy fn-ptr field
    for _ in range(3):
        var h2 = arc_h.copy()
        keep(h2.length())
    t0 = perf_counter_ns()
    for _ in range(ITERS):
        var h2 = arc_h.copy()
        keep(h2.length())
    var arc_ns = _ns(perf_counter_ns() - t0, ITERS)

    print("C. Copy cost (ns/op,", ITERS, "iters):")
    print("   copy  (AnyArray) :", copy_ns)
    print("   arc  + vtable    :", arc_ns)
    print("   owned+ vtable    : N/A (move-only)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() raises:
    print("=== Type Erasure Benchmark (int64, length=1 000 000) ===")
    var arr = _make_array(1_000_000)
    bench_a_erase(arr)
    print()
    bench_b_dispatch(arr)
    print()
    bench_c_copy(arr)
