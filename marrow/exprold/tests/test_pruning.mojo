"""Statistics-based predicate pruning (marrow.exprold.pruning). A predicate is
evaluated against per-column [min,max] bounds and must return maybe_true=False
only when it provably cannot match. Covers both the runtime DynValue interpreter
and the fused static nodes, plus the `BoxedValue` box the scan uses."""

from std.testing import assert_true, assert_false
from ... import dtypes as dt
from ...dtypes import int64, Int64Type, Field
from ...schema import Schema
from ...scalars import DynScalar, Int64Scalar
from ...exprold.pruning import PruneStats
from ...exprold.values import BoxedValue
from ...scalars import StringScalar
from ...exprold.dynamic import DynValue
from ...exprold.values import StrGt
from ...exprold.builders import col, lit
from ...exprold.builders import col as dyn_col

# Both lanes are covered: the runtime `DynValue` cases first, the fused
# `marrow.exprold.values` cases below. A previous note here claimed fused pruning
# was "PARKED" and the per-node overrides unported — that was false; they are at
# `values.mojo` on `NumericColumn`, `NumericLiteral`, `NumericCompare` and
# `BoolBinary`. Only the *tests* were missing.


def _stats(xmin: Int, xmax: Int, ymin: Int, ymax: Int) raises -> PruneStats:
    """Two int64 columns x, y with the given [min,max] bounds."""
    var fields = List[Field]()
    fields.append(Field("x", int64))
    fields.append(Field("y", int64))
    var mins = List[Optional[DynScalar]]()
    var maxs = List[Optional[DynScalar]]()
    mins.append(Optional[DynScalar](Int64Scalar(Int64(xmin))))
    maxs.append(Optional[DynScalar](Int64Scalar(Int64(xmax))))
    mins.append(Optional[DynScalar](Int64Scalar(Int64(ymin))))
    maxs.append(Optional[DynScalar](Int64Scalar(Int64(ymax))))
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
# Through the BoxedValue box (what the scan holds)
# ---------------------------------------------------------------------------


def test_boxed_dyn() raises:
    var boxed_dyn = BoxedValue(col("x") > lit[Int64Type](Int64(100)))
    assert_false(boxed_dyn.prune(_stats(0, 50, 0, 0)).maybe_true)
    assert_true(boxed_dyn.prune(_stats(0, 200, 0, 0)).maybe_true)


def test_unknown_stats_keeps() raises:
    # a column with no stats (None bounds) must never be pruned
    var fields = List[Field]()
    fields.append(Field("x", int64))
    var mins = List[Optional[DynScalar]]()
    var maxs = List[Optional[DynScalar]]()
    mins.append(None)
    maxs.append(None)
    var stats = PruneStats(Schema(fields=fields^), mins^, maxs^)
    var pred = col("x") > lit[Int64Type](Int64(100))
    assert_true(pred.prune(stats).maybe_true)


# ---------------------------------------------------------------------------
# Fused lane.
#
# The per-node `prune` overrides *are* present in `marrow.exprold.values`
# (`NumericColumn`, `NumericLiteral`, `NumericCompare`, `BoolBinary`), but every
# case above builds a runtime `DynValue` predicate, so the fused path had no
# coverage at all. A typo in `NumericCompare.prune`'s comptime switch on
# `Self.K.name` falls through to `unknown()`, which is *sound* — it just stops
# pruning silently, so nothing fails.
# ---------------------------------------------------------------------------


def test_fused_gt_literal() raises:
    var pred = col("x", int64) > lit(100, int64)
    assert_false(pred.prune(_stats(0, 50, 0, 0)).maybe_true)
    assert_true(pred.prune(_stats(0, 200, 0, 0)).maybe_true)


def test_fused_lt_literal() raises:
    var pred = col("x", int64) < lit(10, int64)
    assert_false(pred.prune(_stats(20, 30, 0, 0)).maybe_true)
    assert_true(pred.prune(_stats(0, 30, 0, 0)).maybe_true)


def test_fused_and_prunes_when_either_side_cannot_match() raises:
    var pred = (col("x", int64) > lit(100, int64)) & (
        col("y", int64) < lit(10, int64)
    )
    # x can never exceed 100 -> the conjunction cannot match
    assert_false(pred.prune(_stats(0, 50, 0, 5)).maybe_true)
    # both sides possible -> keep
    assert_true(pred.prune(_stats(0, 200, 0, 5)).maybe_true)


def test_fused_or_keeps_when_either_side_may_match() raises:
    var pred = (col("x", int64) > lit(100, int64)) | (
        col("y", int64) < lit(10, int64)
    )
    assert_true(pred.prune(_stats(0, 50, 0, 5)).maybe_true)
    # neither side can match -> prune
    assert_false(pred.prune(_stats(0, 50, 20, 30)).maybe_true)


def test_boxed_fused_predicate_prunes() raises:
    """The box must delegate `prune` to the fused node, not answer unknown."""
    var boxed: BoxedValue = col("x", int64) > lit(100, int64)
    assert_false(boxed.prune(_stats(0, 50, 0, 0)).maybe_true)
    assert_true(boxed.prune(_stats(0, 200, 0, 0)).maybe_true)


# ---------------------------------------------------------------------------
# A5 — the two lanes must prune the same predicate the same way
# ---------------------------------------------------------------------------
def _string_stats() raises -> PruneStats:
    """One string column `s` bounded to ["m", "p"]."""
    var fields = List[Field]()
    fields.append(Field("s", dt.string))
    var mins = List[Optional[DynScalar]]()
    var maxs = List[Optional[DynScalar]]()
    mins.append(Optional[DynScalar](StringScalar("m")))
    maxs.append(Optional[DynScalar](StringScalar("p")))
    return PruneStats(Schema(fields=fields^), mins^, maxs^)


def test_string_predicate_prunes_in_both_lanes() raises:
    """`s > "z"` cannot match a row group whose `s` maxes out at "p".

    The runtime lane already proves it: `DynValue.prune` keys on
    `_tag == "column"` regardless of dtype, so a string column reports its
    bounds like any other. The fused lane does not — `prune` and
    `bound_column` are defined on `NumericColumn` alone, so `StringColumn`
    inherits `Value`'s conservative default and the predicate prunes nothing.

    That is sound but silently disabled: an AOT plan carrying a string
    predicate decodes every row group, and no test or error says so. The two
    lanes must agree, which is what makes this an A5 case rather than a
    performance note.
    """
    var stats = _string_stats()

    var runtime = dyn_col("s") > DynValue.literal(StringScalar("z"))
    assert_false(
        runtime.prune(stats).maybe_true,
        'runtime lane failed to prune `s > "z"` against max="p"',
    )

    var fused = StrGt(col("s", dt.string), lit("z"))
    assert_false(
        fused.prune(stats).maybe_true,
        'fused lane failed to prune `s > "z"` against max="p"',
    )
