# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Runs: consecutive rows equal on a key.

A **run** is a maximal stretch of consecutive rows that hold the same key.
`RunsKernel` finds where each run starts and answers the positions a `ByEnds`
reads, so the runs are segments: a window's partitions and peer groups, and
the rows whose frames are equal.

"Equal" is `IS NOT DISTINCT FROM`, the key identity `GROUP BY` groups by:
nulls are equal to each other, every NaN is one value, and a nested key
compares structurally.
"""

from ..arrays import BoolArray, DynArray, Int64Array
from ..buffers import Bitmap
from ..builders import arange
from ..dtypes import Int32Type
from ..errors import InvalidError
from ..execution import ExecContext
from .core import Kernel
from .filter import IndicesNonzeroKernel
from .hashing import KeyCompare


struct RunsKernel(Kernel):
    """Where each run of rows equal on every key starts."""

    comptime name = "runs"

    @staticmethod
    def dispatch(
        keys: List[DynArray],
        length: Int,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> Int64Array:
        """The ascending positions where a run starts, then `length`: the
        `count + 1` positions a `ByEnds` reads. `[0]` when `length` is 0, and
        `[0, length]` with no key, since every row is then one run.

        Raises:
            InvalidError: unless every key holds `length` rows.
        """
        for ref key in keys:
            if len(key) != length:
                raise Self.error[InvalidError](
                    t"a key holds {len(key)} rows, not {length}"
                )
        # One bit per row and one past the last: a run starts at row 0, and
        # the last run ends at `length`.
        var starts = Bitmap.alloc_zeroed(length + 1)
        starts.set(0)
        starts.set(length)
        if length > 1:
            # Bit `j` is whether row `j + 1` equals row `j` on every key;
            # each key compares only the pairs the earlier keys kept.
            var same = Bitmap.alloc_zeroed(length - 1)
            same.set_range(0, length - 1, True)
            var rows = arange[Int32Type](0, length - 1)
            for ref key in keys:
                KeyCompare.dispatch(
                    key.slice(1, length - 1),
                    rows,
                    key.slice(0, length - 1),
                    rows,
                    same,
                    ctx,
                )
            var changed = ~same.view(0, length - 1)
            starts.extend(changed.view(0, length - 1), 1, length - 1)
        var positions = IndicesNonzeroKernel.apply(
            BoolArray(
                length=length + 1,
                buffer=starts^.to_immutable(length=length + 1),
            )
        )
        # No position exceeds `length`, so its uint64 bits read as int64.
        return Int64Array(length=len(positions), buffer=positions.buffer)
