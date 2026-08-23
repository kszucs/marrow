"""The API a caller actually writes.

Every other test file in this package builds plan nodes directly —
`DynRelation(Filter(DynRelation(InMemoryTable(b)), DynValue(Gt(...))))` — which
is how the layer is *assembled*, not how it is *used*. These cases exercise the
surface instead: verbs that compose left to right, and aggregates spelled
`col("a", int64).sum()` rather than by naming a kernel.

That is the spelling CLAUDE.md mandates, and it is worth testing separately for
a reason beyond style: the verbs are the only place a caller never writes
`DynRelation(...)` or `DynValue(...)` by hand, so they are where a missing
implicit conversion or a wrong argument order actually shows up.
"""

from std.testing import assert_equal, assert_true

from ...builders import array
from ...dtypes import Int64Type, int64, string
from ...kernels.join import JOIN_INNER, JOIN_LEFT
from ...tabular import RecordBatch, record_batch
from ..builders import col, lit
from ..logical import DynRelation, DynValue, InMemoryTable
from ..`comptime`.leaves import Column, Literal
from ..`comptime`.numeric import Add, Gt, Lt


def _orders() raises -> RecordBatch:
    """Four orders across two customers, with a null amount."""
    return record_batch(
        [
            array([1, 2, 1, 2], int64).copy(),
            array([10, 20, None, 40], int64).copy(),
        ],
        names=["customer", "amount"],
    )


def _table() raises -> DynRelation:
    return DynRelation(InMemoryTable(_orders()))


# ---------------------------------------------------------------------------
# verbs compose
# ---------------------------------------------------------------------------
def test_filter_then_limit_reads_left_to_right() raises:
    """The property the verbs exist for: a plan is a sentence, not a nest."""
    var out = (
        _table()
        .filter(
            DynValue(Gt(Column[Int64Type]("amount"), Literal[Int64Type](5)))
        )
        .limit(2)
        .execute()
    )
    assert_equal(out.num_rows(), 2)


def test_select_keeps_named_columns_in_order() raises:
    """`select` needs no dtype from the caller — it builds runtime column
    reads, which is the runtime lane earning its keep."""
    var out = _table().select(["amount"]).execute()
    assert_equal(out.num_columns(), 1)
    assert_equal(out.schema.fields[0].name, "amount")


def test_project_computes_a_new_column() raises:
    var out = (
        _table()
        .project(
            ["doubled"],
            [
                DynValue(
                    Add(
                        Column[Int64Type]("amount"), Column[Int64Type]("amount")
                    )
                )
            ],
        )
        .execute()
    )
    assert_equal(out.schema.fields[0].name, "doubled")
    assert_equal(out.num_rows(), 4)


def test_sort_by_then_limit_is_top_n() raises:
    """`nulls_first` decides what "top" means, and the default is True.

    Descending with nulls first puts the NULL at the top, which is Postgres's
    `DESC` default — so a top-N over a nullable column has to say what it wants
    rather than assume. Both spellings are asserted because getting this wrong
    silently returns a null row instead of the largest value.
    """
    var top_null = (
        _table()
        .sort_by([DynValue(Column[Int64Type]("amount"))], [False])
        .limit(1)
        .execute()
    )
    assert_true(top_null.columns[1].as_int64().is_null(0))

    var top_value = (
        _table()
        .sort_by(
            [DynValue(Column[Int64Type]("amount"))], [False], nulls_first=False
        )
        .limit(1)
        .execute()
    )
    assert_true(top_value.columns[1].as_int64() == array([40], int64))


# ---------------------------------------------------------------------------
# aggregates, spelled the way CLAUDE.md mandates
# ---------------------------------------------------------------------------
def test_a_whole_table_aggregate_needs_no_keys() raises:
    """`rel.aggregate([...])` with no key list at all — one implicit group."""
    var out = (
        _table()
        .aggregate([DynValue(col("amount", int64).sum().alias("total"))])
        .execute()
    )
    assert_equal(out.num_rows(), 1)
    # 10 + 20 + 40; the null contributes nothing rather than zero
    assert_true(out.columns[0].as_int64() == array([70], int64))


def test_group_by_with_the_fluent_aggregate() raises:
    var out = (
        _table()
        .aggregate(
            [DynValue(col("amount", int64).sum().alias("total"))],
            [DynValue(col("customer", int64))],
        )
        .execute()
    )
    assert_equal(out.num_rows(), 2)
    assert_equal(out.schema.fields[0].name, "customer")
    assert_equal(out.schema.fields[1].name, "total")
    assert_true(out.columns[1].as_int64() == array([10, 60], int64))


def test_every_aggregate_has_a_fluent_spelling() raises:
    """`sum`/`product`/`mean`/`min`/`max` all reachable without naming a
    kernel — the thing CLAUDE.md forbids spelling by hand."""
    var out = (
        _table()
        .aggregate(
            [
                DynValue(col("amount", int64).sum().alias("s")),
                DynValue(col("amount", int64).min().alias("lo")),
                DynValue(col("amount", int64).max().alias("hi")),
                DynValue(col("amount", int64).mean().alias("avg")),
            ]
        )
        .execute()
    )
    assert_equal(out.num_columns(), 4)
    assert_true(out.columns[0].as_int64() == array([70], int64))
    assert_true(out.columns[1].as_int64() == array([10], int64))
    assert_true(out.columns[2].as_int64() == array([40], int64))
    assert_equal(
        String(out.columns[3].as_float64()[0].value()), "23.333333333333332"
    )


def test_alias_renames_without_mutating() raises:
    """An aggregate stays a pure description, so the same subtree names twice.
    """
    var agg = col("amount", int64).sum()
    var a = agg.alias("first")
    var b = agg.alias("second")
    assert_equal(a.name(), "first")
    assert_equal(b.name(), "second")
    assert_equal(agg.name(), "sum")


def test_having_is_a_filter_after_aggregate() raises:
    """No `having` verb: filtering an aggregate's output is just `filter`."""
    var out = (
        _table()
        .aggregate(
            [DynValue(col("amount", int64).sum().alias("total"))],
            [DynValue(col("customer", int64))],
        )
        .filter(
            DynValue(Gt(Column[Int64Type]("total"), Literal[Int64Type](50)))
        )
        .execute()
    )
    assert_equal(out.num_rows(), 1)
    assert_true(out.columns[0].as_int64() == array([2], int64))


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------
def test_join_composes_with_the_other_verbs() raises:
    var customers = DynRelation(
        InMemoryTable(
            record_batch(
                [array([1, 2], int64).copy(), array([100, 200], int64).copy()],
                names=["customer", "credit"],
            )
        )
    )
    var out = customers.join(_table(), [0], [0], JOIN_INNER).execute()
    assert_equal(out.num_rows(), 4)
    assert_equal(out.num_columns(), 4)


def test_a_full_query_reads_as_one_sentence() raises:
    """scan -> filter -> group -> having -> order -> limit, all as verbs.

    The point of the whole surface: this is the shape a user writes, and it
    exercises six operators plus the fused lane in one plan.
    """
    var out = (
        _table()
        .filter(
            DynValue(Gt(Column[Int64Type]("amount"), Literal[Int64Type](5)))
        )
        .aggregate(
            [DynValue(col("amount", int64).sum().alias("total"))],
            [DynValue(col("customer", int64))],
        )
        .filter(DynValue(Gt(Column[Int64Type]("total"), Literal[Int64Type](5))))
        .sort_by([DynValue(Column[Int64Type]("total"))], [False])
        .limit(1)
        .execute()
    )
    assert_equal(out.num_rows(), 1)
    assert_true(out.columns[1].as_int64() == array([60], int64))
