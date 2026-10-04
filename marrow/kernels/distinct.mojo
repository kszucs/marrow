# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Distinct-count kernels — exact and approximate (HyperLogLog).

``count_distinct`` is exact: it is the size of a ``DictionaryEncoder``'s
dictionary. ``approx_count_distinct`` trades exactness for a fixed-size sketch:
a HyperLogLog whose top ``p`` hash bits pick a register and whose remaining
bits' leading-zero run (``rho``) is folded in as a per-register max, matching
``pyarrow.compute.approx_count_distinct``.

Both are whole-array, returning an ``int64`` scalar. The grouped forms behind
the ``count_distinct`` / ``approx_count_distinct`` aggregates are
``DistinctCount`` and ``ApproxDistinctCount`` in ``aggregate.mojo``; the
latter shares the HyperLogLog primitives here.

Both exclude nulls — SQL ``COUNT(DISTINCT x)`` semantics, PyArrow's ``only_valid``.
"""

import std.math as math
from std.bit import count_leading_zeros

from ..arrays import DynArray
from ..dtypes import DynType
from ..scalars import Int64Scalar
from ..execution import ExecContext
from .dictionary import DictionaryEncoder
from .hashing import HashKernel
from ..utils import Fmix64


# ---------------------------------------------------------------------------
# HyperLogLog primitives — shared by the whole-array and per-group estimators,
# parameterized by precision `p` (2**p registers).
# ---------------------------------------------------------------------------


@always_inline
def hll_rho[p: Int](h: UInt64) -> UInt8:
    """Register increment for hash ``h``: 1 + leading-zero run of the bits below
    the top ``p`` (a sentinel bit ORed in caps it at ``64 - p + 1``)."""
    var w = (h << UInt64(p)) | (UInt64(1) << UInt64(p - 1))
    return UInt8(count_leading_zeros(w) + 1)


def hll_estimate[p: Int](registers: List[UInt8], base: Int) -> Int64:
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

comptime HLL_P_GROUPED = 11
"""Per-group HyperLogLog precision: 2**11 = 2048 registers (2 KiB/group,
~2.3% standard error) — bounds memory when the group count is large.

Public, with `hll_rho` and `hll_estimate`, because the streaming aggregate in
`aggregate.mojo` keeps registers of this width between morsels and finalises
them itself. The sketch is the shared thing; the loop over morsels is not."""


# ---------------------------------------------------------------------------
# Whole-array
# ---------------------------------------------------------------------------


def count_distinct(
    array: DynArray, ctx: ExecContext = ExecContext.serial()
) raises -> Int64Scalar:
    """Exact count of distinct non-null values (SQL ``COUNT(DISTINCT x)``,
    PyArrow's ``only_valid``).

    The size of a ``DictionaryEncoder``'s dictionary over the column — exact,
    and partition-parallel under a parallel ``ctx``, since the encoder places a
    large and distinct column on radix-partitioned tables. NULL is one key to
    the encoder, so it is taken off when present.
    """
    var types = List[DynType]()
    types.append(array.dtype())
    var encoder = DictionaryEncoder(types^, ctx.copy())
    var columns = List[DynArray]()
    columns.append(array.copy())
    _ = encoder.encode(columns)
    var n = len(encoder)
    if array.null_count() > 0:
        n -= 1
    return Int64Scalar(Int64(n))


def approx_count_distinct(
    array: DynArray, ctx: ExecContext = ExecContext.serial()
) raises -> Int64Scalar:
    """Approximate count of distinct non-null values via HyperLogLog.

    A fixed 16 KiB sketch (2**14 registers) estimates cardinality with ~0.65%
    standard error independent of the input size — the trade for
    ``count_distinct`` when an exact hash set would be too large. Nulls excluded.
    """
    comptime p = _HLL_P
    comptime m = 1 << p
    var registers = List[UInt8](length=m, fill=0)

    var hv = HashKernel[Fmix64].dispatch(array, ctx).values()
    var n = len(array)
    var has_null = array.null_count() > 0

    for i in range(n):
        if has_null and not array.is_valid(i):
            continue
        var h = UInt64(hv[i])
        var idx = Int(h >> (64 - p))
        var rho = hll_rho[p](h)
        if rho > registers[idx]:
            registers[idx] = rho

    return Int64Scalar(hll_estimate[p](registers, 0))
