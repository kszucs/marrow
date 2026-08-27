"""The runtime lane's aggregates — a name and an erased operand.

`Aggregate` covers the case where the aggregate is written in Mojo and
its operand can stay fused. This covers the other one: the aggregate arrives as
a **string**, from a frontend that built the whole query after the program
started, so both the aggregate and its operands are erased.

The claim under test is that both roads reach the same `AggKernel`, and
that `RuntimeAggregate` refuses a name it cannot serve at the point it is
written rather than on the first morsel.
"""

from std.testing import assert_equal, assert_true

from ....builders import array
from ....dtypes import DynType, int64, string
from ....tabular import RecordBatch, record_batch
from ...builders import col, table
from ...logical import DynValue, Shape
from ..aggregates import RuntimeAggregate
from ..values import column


def _batch() raises -> RecordBatch:
    var values: List[Optional[String]] = ["a", "b", "a", "c", "c"]
    return record_batch(
        [array([1, 1, 1, 2, 2], int64).to_dyn(), array(values).to_dyn()],
        names=["g", "s"],
    )


def test_named_aggregate_rejects_an_unknown_name_where_it_is_written() raises:
    """Validation in `__init__` is why the name is not just a `String` field:
    `"summ"` fails where the node is built, not on the first morsel of a long
    scan.

    Constructed directly rather than through the fluent surface, because the
    fluent surface cannot produce a bad name — which is the point of it.
    """
    var raised = False
    try:
        _ = RuntimeAggregate(DynValue(column("s")), String("summ"))
    except:
        raised = True
    assert_true(raised)
    # And a real one does not.
    _ = RuntimeAggregate(DynValue(column("s")), String("count_distinct"))


def test_named_aggregate_resolution_answers_dtype_and_fold_together() raises:
    """One ladder, so the schema dtype and the implementation cannot disagree.

    Both come off the same `AggKernel` on the same branch — which is
    what replaces the compiler check `Aggregation.dtype` used to get from
    having to match `grouped`'s return type.
    """
    var dtypes = List[DynType](capacity=1)
    dtypes.append(DynType(string))
    var counted = col("s").count_distinct().resolve(dtypes)
    assert_true(counted.dtype == DynType(int64))

    var extremum = col("s").min().resolve(dtypes)
    assert_true(extremum.dtype == DynType(string))


def test_named_aggregate_is_scalar_shaped_and_named_by_its_function() raises:
    """Built through the fluent surface — `col("s")` with no dtype is the
    runtime lane, so the same spelling that fuses in the comptime lane lands
    here instead."""
    var agg = col("s").count_distinct()
    assert_equal(agg.name(), "count_distinct")
    assert_equal(agg.shape, Shape.scalar)
    assert_equal(String(agg), "count_distinct(s)")
    assert_equal(agg.columns()[0], "s")


def test_named_aggregate_alias_leaves_the_function_alone() raises:
    """Two name fields. `_alias` reaches the output schema; `_func` is what
    resolves, and `alias` must not touch it — one field would print `n(col(s))`
    and send the resolver looking for an aggregate called `n`."""
    var agg = col("s").count_distinct().alias("n")
    assert_equal(agg.name(), "n")
    assert_equal(String(agg), "count_distinct(s)")


def test_named_aggregate_runs_keyless() raises:
    """The erased operand resolves against the batch's schema, and the empty
    id array takes the whole-input branch."""
    var plan = table(_batch()).aggregate(
        [col("s").count_distinct()], List[DynValue]()
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 1)
    assert_true(out.columns[0].as_int64() == array([3], int64))
    assert_true(plan.schema() == out.schema)


def test_named_aggregate_runs_grouped() raises:
    """`col("g")` is a runtime key too — the whole plan is built from names."""
    var plan = table(_batch()).aggregate(
        [col("s").min().alias("lo")], [col("g")]
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 2)
    assert_true(out.columns[1].as_string() == array(["a", "c"]))
    assert_true(plan.schema() == out.schema)


def test_named_aggregate_covers_the_folds_the_comptime_lane_fuses() raises:
    """`sum`/`mean`/`count` are reachable by name as well as by type.

    The comptime lane fuses these into `Aggregate`; a frontend that only
    has a string reaches the same algebra through `Fold` and
    `ValidCount` instead. Same answers, one materialised column.
    """
    var plan = table(_batch()).aggregate(
        [
            col("g").sum().alias("total"),
            col("g").mean().alias("avg"),
            col("s").count().alias("n"),
        ],
        List[DynValue](),
    )
    var out = plan.execute()
    assert_true(out.columns[0].as_int64() == array([7], int64))
    assert_equal(out.columns[1].as_float64()[0].value(), 1.4)
    assert_true(out.columns[2].as_int64() == array([5], int64))
    assert_true(plan.schema() == out.schema)
