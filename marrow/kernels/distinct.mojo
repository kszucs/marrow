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
latter shares ``HLL_P`` with the whole-array estimate.

Both exclude nulls — SQL ``COUNT(DISTINCT x)`` semantics, PyArrow's ``only_valid``.
"""

from ..arrays import DynArray
from ..dtypes import DynType
from ..scalars import Int64Scalar
from ..execution import ExecContext
from .core import Kernel
from .dictionary import DictionaryEncoder
from .hashing import HashKernel
from ..utils import Fmix64, HyperLogLog


comptime HLL_P = 14
"""Single-sketch HyperLogLog precision: 2**14 = 16384 registers, ~0.65%
error -- a whole array, or an aggregate with no grouping."""


# ---------------------------------------------------------------------------
# Whole-array
# ---------------------------------------------------------------------------


struct CountDistinctKernel(Kernel):
    """Exact count of distinct non-null values (SQL ``COUNT(DISTINCT x)``,
    PyArrow's ``only_valid``).

    The size of a ``DictionaryEncoder``'s dictionary over the column — exact,
    and partition-parallel under a parallel ``ctx``, since the encoder places a
    large and distinct column on radix-partitioned tables. NULL is one key to
    the encoder, so it is taken off when present.
    """

    comptime name = "count_distinct"

    @staticmethod
    def dispatch(
        array: DynArray, ctx: ExecContext = ExecContext.serial()
    ) raises -> Int64Scalar:
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


struct ApproxCountDistinctKernel(Kernel):
    """Approximate count of distinct non-null values via HyperLogLog.

    A fixed 16 KiB sketch (2**14 registers) estimates cardinality with ~0.65%
    standard error independent of the input size — the trade for
    ``count_distinct`` when an exact hash set would be too large. Nulls excluded.
    """

    comptime name = "approx_count_distinct"

    @staticmethod
    def dispatch(
        array: DynArray, ctx: ExecContext = ExecContext.serial()
    ) raises -> Int64Scalar:
        var sketch = HyperLogLog[HLL_P](1)

        var hv = HashKernel[Fmix64].dispatch(array, ctx).values()
        var n = len(array)
        var has_null = array.null_count() > 0

        for i in range(n):
            if has_null and not array.is_valid(i):
                continue
            sketch.add(0, UInt64(hv[i]))

        return Int64Scalar(sketch.estimate(0))
