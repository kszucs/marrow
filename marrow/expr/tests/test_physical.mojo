"""The physical contract — `Operator`, `Morsel`, `Pipeline`.

These are the tests the plan-building API deliberately *cannot* express.
Everything reachable through `table(...).filter(...)` lives in
`test_relations.mojo`; what lives here is the executor contract itself: that a
streaming stage answers from `push` and a blocking one from `drain`, that
`drain` is repeatable, that `done` lets a source stop early, and that a
`Pipeline` is itself an `Operator` — which is what lets a join hold two whole
sub-plans.

Written against the operators on purpose. `b128b3e` added the `Pipeline`
coverage precisely because it was reachable only indirectly, and a fold that
could not report "spent" was a latent infinite loop that no plan-level test
would have caught.
"""

from std.testing import assert_equal, assert_true
from ...arrays import StructArray, DynArray
from ...builders import array
from ...dtypes import DynType, Int64Type, float64, int64
from ...execution import ExecContext
from ...kernels.join import JOIN_INNER, JOIN_LEFT, JOIN_SEMI
from ...dtypes import field
from ...schema import schema
from ...parquet.writer import write_table
from ...tabular import Table
from ...tabular import RecordBatch, record_batch
from ..logical import DynValue
from ..physical import (
    Datum,
    DynOperator,
    Morsel,
    Pipeline,
)
from ..`comptime`.leaves import Column, Literal
from ..`comptime`.aggregates import Min, Sum
from ..`comptime`.numeric import Add, Gt
from ..builders import col, lit, table
from ..logical import DynValue, DynRelation


def _batch() raises -> RecordBatch:
    """`a` has a null so validity has something to report."""
    return record_batch(
        [
            array([1, 2, None, 4], int64).copy(),
            array([10, 20, 30, 40], int64).copy(),
        ],
        names=["a", "b"],
    )


def test_a_streaming_operator_answers_from_push() raises:
    """`Filter` produces per morsel and has nothing to flush."""
    var op = (
        table(_batch())
        .filter(col("a", int64) > lit(2, int64))
        .to_operator(ExecContext.serial())
    )
    var out = op.drain()
    assert_true(Bool(out))
    assert_equal(len(out.value().struct_array()), 1)
    assert_true(not Bool(op.drain()))


def test_a_blocking_operator_answers_nothing_until_finish() raises:
    """The distinction the push interface replaces a *type* with.

    An aggregate accumulates through every `push` and answers `None`; its whole
    result arrives from `finish`. Under the pull design this needed a different
    trait from `Filter` and therefore a second erased box — collapsing that is
    the point of the engine.
    """
    var op = (
        table(_batch())
        .aggregate([col("a", int64).sum().alias("total")])
        .to_operator(ExecContext.serial())
    )
    var out = op.drain()
    assert_true(Bool(out))
    assert_true(
        out.value().struct_array().children[0].as_int64() == array([7], int64)
    )
    # Emitting the fold twice would double every grouped result downstream.
    assert_true(not Bool(op.drain()))


def test_limit_reports_done_so_the_source_can_stop() raises:
    """The signal a push engine needs and a pull engine gets for free.

    Without it a `LIMIT 10` over a billion-row scan reads a billion rows: the
    source drives, so nothing downstream could otherwise halt it.
    """
    var op = (
        table(_batch())
        .limit(length=2, offset=0)
        .to_operator(ExecContext.serial())
    )
    assert_true(not op.done())
    var out = op.drain()
    assert_true(Bool(out))
    assert_equal(len(out.value().struct_array()), 2)
    assert_true(op.done())


def test_sort_then_limit_is_top_n() raises:
    """The composition every engine needs, and the one that proves a blocking
    stage and a bounded stage cooperate: `Sort` emits only from `finish`, and
    the `Limit` above it must still see that batch through the flush cascade.
    """
    var b = record_batch([array([5, 3, 9, 1], int64).copy()], names=["a"])
    var plan = (
        table(b^).sort_by([col("a", int64)], [True]).limit(length=2, offset=0)
    )
    assert_true(plan.execute().columns[0].as_int64() == array([1, 3], int64))


# ---------------------------------------------------------------------------
# Pipeline — a composite Operator
# ---------------------------------------------------------------------------


def test_a_pipeline_drains_until_spent() raises:
    """`drain` is resumable and must eventually answer `None`.

    The driver loops on it, so a pipeline that never reported spent would hang.
    Asserted directly because `collect` hides it: `collect` stops at the first
    `None` and would look correct even if the second call answered `Some`
    again.
    """
    var pipe = table(_batch()).to_operator(ExecContext.serial())
    var first = pipe.drain()
    assert_true(Bool(first))
    assert_equal(len(first.value().struct_array()), 4)
    assert_true(not Bool(pipe.drain()))
    assert_true(not Bool(pipe.drain()))


def test_a_pipeline_cascades_through_its_stages() raises:
    """A batch leaving stage `i` must be seen by `i+1..` before `i+1` drains.

    Here the source's batch has to reach the filter, which is the same cascade
    an aggregate under a projection depends on — checked at the `Pipeline`
    level rather than only through `execute`.
    """
    var pipe = (
        table(_batch())
        .filter(col("a", int64) > lit(2, int64))
        .to_operator(ExecContext.serial())
    )
    var got = pipe.drain()
    assert_true(Bool(got))
    assert_equal(
        len(got.value().struct_array()), 1
    )  # a = [1, 2, None, 4] -> only 4
    assert_true(not Bool(pipe.drain()))


def test_a_pipeline_is_an_operator() raises:
    """It is a composite, not a second abstraction — so it boxes like any
    other stage. This is what lets `Join` hold two whole sub-plans."""
    var pipe = table(_batch()).to_operator(ExecContext.serial())
    var boxed = DynOperator(pipe^)
    var got = boxed.drain()
    assert_true(Bool(got))
    assert_equal(len(got.value().struct_array()), 4)


# ---------------------------------------------------------------------------
# Join — the operator with two inputs
# ---------------------------------------------------------------------------
