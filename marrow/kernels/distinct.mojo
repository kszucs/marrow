"""Distinct-count kernels — exact and approximate (HyperLogLog).

``count_distinct`` is exact to the same 64-bit-hash basis the group-by hash
table dedups on (collision probability ~n^2/2^64). ``approx_count_distinct``
trades exactness for a fixed-size sketch: a HyperLogLog with 2**14 registers
(~0.65% standard error, 16 KiB regardless of cardinality), mirroring
``pyarrow.compute.approx_count_distinct`` and SQL engines' ``approx_count_distinct``.

Both exclude nulls — SQL ``COUNT(DISTINCT x)`` semantics, PyArrow's ``only_valid``.
"""

import std.math as math
from std.bit import count_leading_zeros

from ..arrays import AnyArray
from ..scalars import Int64Scalar
from .execution import ExecutionContext
from .hashing import rapidhash
from .hashtable import SwissHashTable


def count_distinct(
    array: AnyArray, ctx: ExecutionContext = ExecutionContext.serial()
) raises -> Int64Scalar:
    """Exact count of distinct non-null values.

    Dedups the per-row hashes through the same ``SwissHashTable`` the group-by
    uses, so it is exact to that 64-bit-hash basis (collision probability
    ~n^2/2^64 — the same basis group-by itself dedups on). Nulls are excluded
    (SQL ``COUNT(DISTINCT x)`` / PyArrow ``only_valid``): every null hashes to a
    single sentinel bucket, subtracted off when the array has any null.
    """
    var hashes = rapidhash(array, ctx)
    var table = SwissHashTable[rapidhash]()
    _ = table.insert_hashes(hashes, grow_adaptively=True)
    var n = table.num_keys()
    if array.null_count() > 0:
        n -= 1  # drop the single sentinel bucket every null collapsed into
    return Int64Scalar(Int64(n))


comptime _HLL_P = 14
"""HyperLogLog precision: 2**14 = 16384 registers → ~0.65% standard error."""


def approx_count_distinct(
    array: AnyArray, ctx: ExecutionContext = ExecutionContext.serial()
) raises -> Int64Scalar:
    """Approximate count of distinct non-null values via HyperLogLog.

    A fixed 16 KiB sketch (2**14 registers) estimates cardinality with ~0.65%
    standard error independent of the input size — the trade for
    ``count_distinct`` when an exact hash set would be too large. Uses 64-bit
    hashes with linear counting in the small-cardinality regime; the 32-bit
    large-range correction is unnecessary. Nulls are excluded.
    """
    comptime p = _HLL_P
    comptime m = 1 << p
    var registers = List[UInt8](length=m, fill=0)

    var hashes = rapidhash(array, ctx)
    var hv = hashes.values()
    var n = len(array)
    var has_null = array.null_count() > 0

    for i in range(n):
        if has_null and not array.is_valid(i):
            continue
        var h = UInt64(hv[i])
        # Top p bits pick the register; the leading-zero run of the remaining
        # bits (with a sentinel bit ORed in to cap rho at 64-p+1) is rho.
        var idx = Int(h >> (64 - p))
        var w = (h << p) | (UInt64(1) << (p - 1))
        var rho = UInt8(count_leading_zeros(w) + 1)
        if rho > registers[idx]:
            registers[idx] = rho

    # Harmonic-mean (raw) estimate with the standard bias constant.
    var inv_sum = Float64(0)
    var zeros = 0
    for j in range(m):
        var r = Int(registers[j])
        inv_sum += Float64(1) / Float64(UInt64(1) << UInt64(r))
        if r == 0:
            zeros += 1
    var alpha = 0.7213 / (1 + 1.079 / Float64(m))
    var estimate = alpha * Float64(m) * Float64(m) / inv_sum
    # Linear counting for small cardinalities (many still-empty registers).
    if estimate <= 2.5 * Float64(m) and zeros > 0:
        estimate = Float64(m) * math.log(Float64(m) / Float64(zeros))
    return Int64Scalar(Int64(Int(estimate + 0.5)))
