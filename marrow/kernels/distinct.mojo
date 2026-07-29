"""Distinct-count kernels — exact and approximate (HyperLogLog).

``count_distinct`` is exact to the same 64-bit-hash basis the group-by hash
table dedups on (collision probability ~n^2/2^64). ``approx_count_distinct``
trades exactness for a fixed-size sketch: a HyperLogLog whose top ``p`` hash
bits pick a register and whose remaining bits' leading-zero run (``rho``) is
folded in as a per-register max, matching ``pyarrow.compute.approx_count_distinct``.

Both come in a whole-array form (returns an ``int64`` scalar) and a **grouped**
form (``*_grouped(gids, value, num_groups)`` → one ``int64`` per group), the
latter driving ``GroupBy``'s ``count_distinct`` / ``approx_count_distinct``:

- exact grouped dedups ``(group_id, value)`` pairs in a single ``SwissHashTable``
  (the join's table) and bumps a per-group counter on each newly-seen pair — one
  pass, no per-group set.
- approx grouped keeps one HyperLogLog sketch per group in a flat register array.

Both exclude nulls — SQL ``COUNT(DISTINCT x)`` semantics, PyArrow's ``only_valid``.
"""

import std.math as math
from std.bit import count_leading_zeros

from ..arrays import DynArray, Int32Array, Int64Array, UInt64Array, StructArray
from ..builders import Int64Builder
from ..dtypes import Field, int32, struct_
from ..scalars import Int64Scalar
from .execution import ExecutionContext
from .hashing import rapidhash
from .hashtable import SwissHashTable
from .partition import RadixPartitioner


comptime _PARALLEL_DISTINCT_MIN_ROWS = 200_000
"""Below this the serial hash-set dedup wins — radix partition + thread dispatch
overhead would dominate."""


# ---------------------------------------------------------------------------
# HyperLogLog primitives — shared by the whole-array and per-group estimators,
# parameterized by precision `p` (2**p registers).
# ---------------------------------------------------------------------------


@always_inline
def _hll_rho[p: Int](h: UInt64) -> UInt8:
    """Register increment for hash ``h``: 1 + leading-zero run of the bits below
    the top ``p`` (a sentinel bit ORed in caps it at ``64 - p + 1``)."""
    var w = (h << UInt64(p)) | (UInt64(1) << UInt64(p - 1))
    return UInt8(count_leading_zeros(w) + 1)


def _hll_estimate[p: Int](registers: List[UInt8], base: Int) -> Int64:
    """Cardinality estimate for the ``2**p`` registers at ``registers[base:]``.

    Harmonic-mean (raw) estimate with the standard bias constant, falling back
    to linear counting when many registers are still empty. 64-bit hashes make
    the 32-bit large-range correction unnecessary."""
    comptime m = 1 << p
    var inv_sum = Float64(0)
    var zeros = 0
    for j in range(m):
        var r = Int(registers[base + j])
        inv_sum += Float64(1) / Float64(UInt64(1) << UInt64(r))
        if r == 0:
            zeros += 1
    var alpha = 0.7213 / (1 + 1.079 / Float64(m))
    var estimate = alpha * Float64(m) * Float64(m) / inv_sum
    if estimate <= 2.5 * Float64(m) and zeros > 0:
        estimate = Float64(m) * math.log(Float64(m) / Float64(zeros))
    return Int64(Int(estimate + 0.5))


comptime _HLL_P = 14
"""Whole-array HyperLogLog precision: 2**14 = 16384 registers → ~0.65% error."""

comptime _HLL_P_GROUPED = 11
"""Per-group HyperLogLog precision: 2**11 = 2048 registers (2 KiB/group,
~2.3% standard error) — bounds memory when the group count is large."""


# ---------------------------------------------------------------------------
# Whole-array
# ---------------------------------------------------------------------------


def count_distinct(
    array: DynArray, ctx: ExecutionContext = ExecutionContext.serial()
) raises -> Int64Scalar:
    """Exact count of distinct non-null values.

    Dedups the per-row hashes through the same ``SwissHashTable`` the group-by
    uses, so it is exact to that 64-bit-hash basis (collision probability
    ~n^2/2^64 — the same basis group-by itself dedups on). Nulls are excluded
    (SQL ``COUNT(DISTINCT x)`` / PyArrow ``only_valid``): every null hashes to a
    single sentinel bucket, subtracted off when the array has any null.

    At scale with a parallel ``ctx`` the dedup is radix-partition-parallel: a
    value hashes to exactly one partition, so distinct values are split disjointly
    and the total is the *sum* of per-partition distinct counts — no merge, the
    whole-array analogue of the grouped radix path.
    """
    var hashes = rapidhash(array, ctx)
    var nt = ctx.resolved_num_threads()
    var n: Int
    if nt <= 1 or len(array) < _PARALLEL_DISTINCT_MIN_ROWS:
        var table = SwissHashTable[rapidhash]()
        _ = table.insert_hashes(hashes, grow_adaptively=True)
        n = table.num_keys()
    else:

        @parameter
        def count_partition(
            _pi: Int, _rows: Int32Array, part_hashes: UInt64Array
        ) raises -> Int:
            var table = SwissHashTable[rapidhash]()
            _ = table.insert_hashes(part_hashes, grow_adaptively=True)
            return table.num_keys()

        var counts = RadixPartitioner(
            num_bits=6, ctx=ctx.copy()
        ).map_partitions[Int, count_partition](hashes^)
        n = 0
        for i in range(len(counts)):
            n += counts[i]
    if array.null_count() > 0:
        n -= (
            1  # the single sentinel bucket every null collapsed into (one part)
        )
    return Int64Scalar(Int64(n))


def approx_count_distinct(
    array: DynArray, ctx: ExecutionContext = ExecutionContext.serial()
) raises -> Int64Scalar:
    """Approximate count of distinct non-null values via HyperLogLog.

    A fixed 16 KiB sketch (2**14 registers) estimates cardinality with ~0.65%
    standard error independent of the input size — the trade for
    ``count_distinct`` when an exact hash set would be too large. Nulls excluded.
    """
    comptime p = _HLL_P
    comptime m = 1 << p
    var registers = List[UInt8](length=m, fill=0)

    var hv = rapidhash(array, ctx).values()
    var n = len(array)
    var has_null = array.null_count() > 0

    for i in range(n):
        if has_null and not array.is_valid(i):
            continue
        var h = UInt64(hv[i])
        var idx = Int(h >> (64 - p))
        var rho = _hll_rho[p](h)
        if rho > registers[idx]:
            registers[idx] = rho

    return Int64Scalar(_hll_estimate[p](registers, 0))


# ---------------------------------------------------------------------------
# Grouped — one distinct-count per group id (driven by GroupBy)
# ---------------------------------------------------------------------------


def count_distinct_grouped(
    gids: Int32Array,
    value: DynArray,
    num_groups: Int,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> Int64Array:
    """Exact distinct count of ``value`` per group, over precomputed ``gids``.

    Dedups ``(group_id, value)`` pairs in one ``SwissHashTable``: each pair's
    combined hash is inserted once, and the first time a pair is seen its group's
    counter is bumped. One pass, O(distinct pairs) memory, no per-group set.
    Null values are excluded.
    """
    var n = len(gids)
    # Hash the (group_id, value) pair per row via the struct hasher (per-field
    # rapidhash + combine) — reusing the exact join/group-by hashing path.
    var children = List[DynArray]()
    children.append(gids.copy())
    children.append(value.copy())
    var pairs = StructArray(
        dtype=struct_(Field("g", int32), Field("v", value.dtype().copy())),
        length=n,
        nulls=0,
        offset=0,
        bitmap=None,
        children=children^,
    )
    var table = SwissHashTable[rapidhash]()
    var bids = table.insert_hashes(rapidhash(pairs, ctx), grow_adaptively=True)

    var seen = List[Bool](length=table.num_keys(), fill=False)
    var counts = List[Int64](length=num_groups, fill=0)
    var has_null = value.null_count() > 0
    for i in range(n):
        if has_null and not value.is_valid(i):
            continue
        var b = Int(bids.unsafe_get(i))
        if not seen[b]:
            seen[b] = True
            counts[Int(gids.unsafe_get(i))] += 1

    var out = Int64Builder(num_groups)
    for g in range(num_groups):
        out.append(counts[g])
    return out.finish()


def approx_count_distinct_grouped(
    gids: Int32Array,
    value: DynArray,
    num_groups: Int,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> Int64Array:
    """Approximate distinct count of ``value`` per group via one HyperLogLog
    sketch per group (2**11 registers each). Bounds memory at
    ``num_groups * 2 KiB`` regardless of per-group cardinality. Nulls excluded.
    """
    comptime p = _HLL_P_GROUPED
    comptime m = 1 << p
    var registers = List[UInt8](length=num_groups * m, fill=0)

    var hv = rapidhash(value, ctx).values()
    var n = len(gids)
    var has_null = value.null_count() > 0
    for i in range(n):
        if has_null and not value.is_valid(i):
            continue
        var h = UInt64(hv[i])
        var idx = Int(gids.unsafe_get(i)) * m + Int(h >> (64 - p))
        var rho = _hll_rho[p](h)
        if rho > registers[idx]:
            registers[idx] = rho

    var out = Int64Builder(num_groups)
    for g in range(num_groups):
        out.append(_hll_estimate[p](registers, g * m))
    return out.finish()
