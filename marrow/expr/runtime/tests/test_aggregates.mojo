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

from ....arrays import StringArray
from ....builders import array
from ....dtypes import DynType, Int64Type, float64, int32, int64, string
from ....dtypes import StringType
from ....kernels.aggregate import (
    AggKernel,
    Dispersion,
    DistinctCount,
    Fold,
    MaxFold,
    MaxOp,
    MeanFold,
    MinFold,
    MinOp,
    ProductFold,
    LexicalExtremum,
    SumFold,
    ValidCount,
)
from ....tabular import RecordBatch, record_batch
from ...builders import col, table
from ...logical import DynValue, Shape
from ..aggregates import RuntimeAggregate, resolve_aggregate
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
        _ = RuntimeAggregate(column("s"), String("summ"))
    except:
        raised = True
    assert_true(raised)
    # And a real one does not.
    _ = RuntimeAggregate(column("s"), String("count_distinct"))


def _out_dtype(name: String, in_dtype: DynType) raises -> DynType:
    """The catalog's answer, through the one ladder both callers use."""

    def job[Agg: AggKernel]() raises {imm} -> DynType:
        return Agg.dtype(in_dtype)

    return resolve_aggregate(name, in_dtype, job)


def test_named_aggregate_resolution_answers_dtype_and_fold_together() raises:
    """The catalog's dtype and the kernel that will run cannot disagree.

    `agg_out_dtype` answers from each kernel's own `dtype` static rather than
    restating a constant, so `min` over a string column reports `string`
    because `LexicalExtremum` says so — not because an arm here spells it.
    """
    assert_true(_out_dtype("count_distinct", DynType(string)) == int64)
    assert_true(_out_dtype("count", DynType(string)) == int64)
    assert_true(_out_dtype("min", DynType(string)) == string)
    assert_true(_out_dtype("max", DynType(string)) == string)
    # `sum(int32)` widens; the widening rule is `SumFold`'s, not the
    # catalog's.
    assert_true(_out_dtype("sum", DynType(int32)) == int64)
    assert_true(_out_dtype("variance", DynType(int64)) == float64)


def test_named_aggregate_rejects_a_dtype_it_has_no_arm_for() raises:
    """The catalog is the domain gate: a `sum` over strings raises at plan
    time, where the query was written, not on the first morsel."""
    var raised = False
    try:
        _ = _out_dtype("sum", DynType(string))
    except:
        raised = True
    assert_true(raised)


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
    `ValidCount[StringArray]` instead. Same answers, one materialised column.
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


def test_named_variance_and_stddev_reach_the_composite_accumulator() raises:
    """`variance` is the first aggregate reachable by name whose state is not
    a scalar — Welford's (count, mean, M2). Nothing about the runtime lane
    changes for it: a name, one `AggKernel`, one materialised column.

    Group column `g` is [1,1,1,2,2]: population variance 0.24, stddev 0.4899.
    """
    var plan = table(_batch()).aggregate(
        [
            col("g").variance().alias("var_g"),
            col("g").stddev().alias("sd_g"),
        ],
        List[DynValue](),
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 1)
    assert_true(abs(out.columns[0].as_float64()[0].value() - 0.24) < 1e-9)
    assert_true(
        abs(out.columns[1].as_float64()[0].value() - 0.4898979485566356) < 1e-9
    )
    assert_true(plan.schema() == out.schema)


def test_named_aggregate_vocabulary_all_resolves() raises:
    """The two tables must agree: every name `__init__` accepts must be one
    `resolve` can serve.

    They are separate — a string list and a ladder that binds types — so
    nothing but this case connects them. It exists because they *did* drift:
    `variance` and `stddev` reached `resolve` one commit before they reached
    the accept list, and every query naming them raised "unknown aggregate"
    from the constructor.
    """
    for ref name in RuntimeAggregate.vocabulary():
        var node = RuntimeAggregate(column("g"), name.copy())
        # Raises if the ladder has no arm for it; int64 is in every domain.

