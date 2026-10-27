# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Column statistics: what a writer records about a column and an estimator
reads back.

`BoundsKernel` — a column's least and greatest valid value, the bounds a
Parquet writer records per page and per chunk and `analyze` keeps for the
estimator. It folds the column with `MinFold` and `MaxFold` over
`Segmented.reduce`, so a bound is the same `min` and `max` the aggregates
answer.
"""

from std.math import isnan

from ..arrays import PrimitiveArray
from ..dtypes import PrimitiveType
from .aggregate import MaxFold, MinFold
from .core import Kernel
from .segments import BySize, Segmented


struct BoundsKernel(Kernel):
    """The least and the greatest valid value of a column, NaN skipped.

    A bound is read as a proof, and a NaN maximum would prove `x > 5` false,
    so NaN is never a bound. A column with no row, only nulls, or only NaN
    among its valid rows has no bounds.
    """

    comptime name = "bounds"

    @staticmethod
    def apply[
        T: PrimitiveType
    ](values: PrimitiveArray[T]) raises -> Tuple[
        Optional[Scalar[T.native]], Optional[Scalar[T.native]]
    ]:
        """`(least, greatest)` over the valid values, each `None` when there
        is none."""
        var whole = BySize.whole(len(values))
        var low = Segmented[MinFold].reduce(values, whole)
        var high = Segmented[MaxFold].reduce(values, whole)
        var least: Optional[Scalar[T.native]] = None
        var greatest: Optional[Scalar[T.native]] = None
        if len(low) == 1 and low.is_valid(0):
            # `min` skips NaN (minNum), so it answers NaN only when every
            # valid value is NaN.
            var all_nan: Bool
            comptime if T.native.is_floating_point():
                all_nan = isnan(low.unsafe_get(0))
            else:
                all_nan = False
            if not all_nan:
                least = low.unsafe_get(0)
                greatest = high.unsafe_get(0)
        return (least, greatest)
