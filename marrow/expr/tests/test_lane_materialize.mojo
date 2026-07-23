"""Arithmetic over a pipeline-breaker operand routes through the SINGLE fused
`Sub` — no separate materialized binary.

`x - avg(x)` (mean-centering): `avg(x)` is a whole-column reduction (a breaker),
materialized once into the context during `prepare`; then it acts as a fused
splat-leaf, so the subtract fuses over `(x, splat(mean))` — the same
`NumericBinary` as `x - lit`.
"""

from std.testing import assert_true

from marrow.testing import TestSuite
from marrow.builders import array
from marrow.dtypes import int64, float64
from marrow.tabular import record_batch, RecordBatch
from marrow.expr.lane import col, run, Sub, Mean, into_array


def _batch() raises -> RecordBatch:
    return record_batch([array([1, 2, 3, 4], int64).copy()], names=["a"])


def test_mean_centering_via_single_binary() raises:
    # avg([1,2,3,4]) = 2.5 ; x - avg(x) = [-1.5, -0.5, 0.5, 1.5]  (int - float -> float)
    var cv = run(Sub(col(0, int64), Mean(col(0, int64))), _batch())
    assert_true(
        into_array(cv, 4) == array([-1.5, -0.5, 0.5, 1.5], float64).to_any()
    )


def main() raises:
    TestSuite.run[__functions_in_module()]()
