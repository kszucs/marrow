"""Statistics-based predicate pruning (marrow.expr.pruning). A predicate is
evaluated against per-column [min,max] bounds and must return maybe_true=False
only when it provably cannot match. Covers both the runtime DynValue interpreter
and the fused static nodes, plus the AnyValue box the scan uses."""

from std.testing import assert_true, assert_false
from marrow.testing import TestSuite
from marrow import dtypes as dt
from marrow.dtypes import int64, Int64Type, Field
from marrow.schema import Schema
from marrow.scalars import AnyScalar, Int64Scalar
from marrow.expr.pruning import PruneStats
from marrow.expr.values import AnyValue
from marrow.expr.dynamic import col, lit

# NOTE: comptime-node pruning is PARKED in the new `marrow.expr.values` (the
# per-node `prune` overrides were not ported from the old fused algebra; the
# `Value.prune` default now returns "unknown"). Only the runtime `DynValue`
# pruning path is exercised here; the fused column-vs-column pruning test and the
# fused half of the boxed test were removed until comptime pruning is re-ported.


def _stats(xmin: Int, xmax: Int, ymin: Int, ymax: Int) raises -> PruneStats:
    """Two int64 columns x, y with the given [min,max] bounds."""
    var fields = List[Field]()
    fields.append(Field("x", int64))
    fields.append(Field("y", int64))
    var mins = List[Optional[AnyScalar]]()
    var maxs = List[Optional[AnyScalar]]()
    mins.append(Optional[AnyScalar](Int64Scalar(Int64(xmin))))
    maxs.append(Optional[AnyScalar](Int64Scalar(Int64(xmax))))
    mins.append(Optional[AnyScalar](Int64Scalar(Int64(ymin))))
    maxs.append(Optional[AnyScalar](Int64Scalar(Int64(ymax))))
    return PruneStats(Schema(fields=fields^), mins^, maxs^)


# ---------------------------------------------------------------------------
# DynValue predicates
# ---------------------------------------------------------------------------


def test_dyn_gt_literal() raises:
    var pred = col("x") > lit[Int64Type](Int64(100))
    # x in [0, 50] -> can never exceed 100 -> prune
    assert_false(pred.prune(_stats(0, 50, 0, 0)).maybe_true)
    # x in [0, 200] -> might exceed 100 -> keep
    assert_true(pred.prune(_stats(0, 200, 0, 0)).maybe_true)


def test_dyn_eq_literal() raises:
    var pred = col("x") == lit[Int64Type](Int64(5))
    # x in [10, 20] -> 5 not in range -> prune
    assert_false(pred.prune(_stats(10, 20, 0, 0)).maybe_true)
    # x in [0, 8] -> 5 in range -> keep
    assert_true(pred.prune(_stats(0, 8, 0, 0)).maybe_true)


def test_dyn_lt_literal() raises:
    var pred = col("x") < lit[Int64Type](Int64(10))
    # x in [20, 30] -> never below 10 -> prune
    assert_false(pred.prune(_stats(20, 30, 0, 0)).maybe_true)
    assert_true(pred.prune(_stats(0, 30, 0, 0)).maybe_true)


def test_dyn_and() raises:
    # (x > 100) AND (x < 200)
    var pred = (col("x") > lit[Int64Type](Int64(100))) & (
        col("x") < lit[Int64Type](Int64(200))
    )
    # x in [0, 50] -> first conjunct false everywhere -> prune
    assert_false(pred.prune(_stats(0, 50, 0, 0)).maybe_true)
    # x in [120, 180] -> both may hold -> keep
    assert_true(pred.prune(_stats(120, 180, 0, 0)).maybe_true)


def test_dyn_or() raises:
    # (x < 0) OR (x > 1000)
    var pred = (col("x") < lit[Int64Type](Int64(0))) | (
        col("x") > lit[Int64Type](Int64(1000))
    )
    # x in [10, 20] -> neither branch possible -> prune
    assert_false(pred.prune(_stats(10, 20, 0, 0)).maybe_true)
    # x in [10, 2000] -> right branch possible -> keep
    assert_true(pred.prune(_stats(10, 2000, 0, 0)).maybe_true)


# ---------------------------------------------------------------------------
# Through the AnyValue box (what the scan holds)
# ---------------------------------------------------------------------------


def test_boxed_dyn() raises:
    var boxed_dyn = AnyValue(col("x") > lit[Int64Type](Int64(100)))
    assert_false(boxed_dyn.prune(_stats(0, 50, 0, 0)).maybe_true)
    assert_true(boxed_dyn.prune(_stats(0, 200, 0, 0)).maybe_true)


def test_unknown_stats_keeps() raises:
    # a column with no stats (None bounds) must never be pruned
    var fields = List[Field]()
    fields.append(Field("x", int64))
    var mins = List[Optional[AnyScalar]]()
    var maxs = List[Optional[AnyScalar]]()
    mins.append(None)
    maxs.append(None)
    var stats = PruneStats(Schema(fields=fields^), mins^, maxs^)
    var pred = col("x") > lit[Int64Type](Int64(100))
    assert_true(pred.prune(stats).maybe_true)


def main() raises:
    TestSuite.run[__functions_in_module()]()
