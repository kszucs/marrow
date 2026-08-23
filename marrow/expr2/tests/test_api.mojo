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

from ...builders import (
    BoolBuilder,
    Date32Builder,
    Int64Builder,
    ListBuilder,
    Time32Builder,
    array,
)
from ...dtypes import (
    Int64Type,
    second,
    date32,
    int64,
    list_,
    string,
    time32,
)
from ...kernels.join import JOIN_INNER, JOIN_LEFT
from ...tabular import RecordBatch, record_batch
from ..builders import array_length, col, if_else, lit
from ..runtime.values import case_when, coalesce, column, literal
from ...scalars import DynScalar, Int64Scalar
from ..logical import DynRelation, DynValue, InMemoryTable
from ..`comptime`.leaves import Column, Literal
from ..`comptime`.numeric import Add, Gt, Lt, TemporalGt


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
    """Scan, filter, group, having, order, limit -- all as verbs.

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


# ---------------------------------------------------------------------------
# conditionals and lists
# ---------------------------------------------------------------------------
def test_if_else_selects_per_row() raises:
    """A null condition counts as **false**, not as a null result.

    That is Arrow's `ExecArrayCaseWhen` rule and PyArrow's `pc.case_when`, and
    it is the reason the node calls the kernel rather than fusing a lane: the
    three-way rule over condition validity is the kernel's.
    """
    var out = (
        _table()
        .project(
            ["capped"],
            [
                DynValue(
                    if_else(
                        Gt(Column[Int64Type]("amount"), Literal[Int64Type](15)),
                        Literal[Int64Type](99),
                        Column[Int64Type]("amount"),
                    )
                )
            ],
        )
        .execute()
    )
    ref got = out.columns[0].as_int64()
    assert_equal(got[0].value(), 10)  # 10 > 15 false -> amount
    assert_equal(got[1].value(), 99)  # 20 > 15 true  -> 99
    # amount is NULL here, so the condition is null -> false -> else branch,
    # and the else branch is itself null.
    assert_true(got.is_null(2))
    assert_equal(got[3].value(), 99)


def test_array_length_consumes_a_list_into_a_number() raises:
    """The shape every list operation takes.

    `ListValue` declares no `lane` because a list element is a whole
    sub-array; operations over it produce a *different* family's lane, and
    `array_length` is the smallest example.
    """
    var ints = Int64Builder()
    var lists = ListBuilder(ints^)
    var child_any = lists.values()
    ref child = child_any.as_int64()
    child.append(1)
    child.append(2)
    child.append(3)
    lists.append_valid()  # [1, 2, 3]
    child.append(4)
    lists.append_valid()  # [4]
    lists.append_null()  # null
    var batch = record_batch([lists.finish().to_dyn()], names=["xs"])

    var out = (
        DynRelation(InMemoryTable(batch^))
        .project(["n"], [DynValue(array_length(col("xs", list_(int64))))])
        .execute()
    )
    ref got = out.columns[0].as_int32()
    assert_equal(got[0].value(), 3)
    assert_equal(got[1].value(), 1)
    assert_true(got.is_null(2))  # a null list has no length


def test_a_temporal_column_can_be_filtered() raises:
    """`WHERE d > <date>` in the fused lane — what S21 blocked.

    A separate node from `NumericCompare` because `promote` encodes numeric
    widening, and the temporal question is *unit* rather than width. This one
    compares only operands that already share a representation.
    """
    var d = Date32Builder(date32(), 3)
    d.append(Int32(19000))
    d.append(Int32(19005))
    d.append_null()
    var e = Date32Builder(date32(), 3)
    e.append(Int32(19002))
    e.append(Int32(19002))
    e.append(Int32(19002))
    var batch = record_batch(
        [d.finish().to_dyn(), e.finish().to_dyn()], names=["d", "cutoff"]
    )

    var out = (
        DynRelation(InMemoryTable(batch^))
        .filter(
            DynValue(TemporalGt(col("d", date32()), col("cutoff", date32())))
        )
        .execute()
    )
    # 19000 > 19002 false; 19005 > 19002 true; the null does not select
    assert_equal(out.num_rows(), 1)


def test_temporal_comparison_rejects_mismatched_units() raises:
    """Same width is not the same type.

    `date32` and `time32[s]` are both int32, so a width check cannot separate
    them — the dtypes are compared at `bind`, once per batch. Cross-unit
    comparison raises rather than silently comparing raw integers.
    """
    var d = Date32Builder(date32(), 1)
    d.append(Int32(19000))
    var t = Time32Builder(time32(second), 1)
    t.append(Int32(19000))
    var batch = record_batch(
        [d.finish().to_dyn(), t.finish().to_dyn()], names=["d", "t"]
    )

    var raised = False
    try:
        _ = (
            DynRelation(InMemoryTable(batch^))
            .filter(
                DynValue(
                    TemporalGt(col("d", date32()), col("t", time32(second)))
                )
            )
            .execute()
        )
    except e:
        raised = True
        assert_true("units must match" in String(e))
    assert_true(raised)


def test_coalesce_takes_the_first_non_null() raises:
    """N-ary, because the kernel is — `expr/` folds binary nodes only because
    its runtime node could not hold N children."""
    var b = record_batch(
        [
            array([1, None, None], int64).copy(),
            array([None, 20, None], int64).copy(),
            array([300, 300, 300], int64).copy(),
        ],
        names=["a", "b", "c"],
    )
    var out = (
        DynRelation(InMemoryTable(b^))
        .project(
            ["first"],
            [DynValue(coalesce([column("a"), column("b"), column("c")]))],
        )
        .execute()
    )
    assert_true(out.columns[0].as_int64() == array([1, 20, 300], int64))


def test_case_when_picks_the_first_true_branch() raises:
    """Arity comes from the child count's **parity**: `2n` without an else,
    `2n + 1` with one. That is why it needs no payload — `EvalFn` is a `thin`
    pointer and cannot capture a flag.

    Conditions are bool columns rather than comparisons because the runtime
    lane has only `column` and `literal` today; it cannot build a predicate of
    its own. That gap is real and separate from this node.
    """
    var lo = BoolBuilder(3)
    lo.append(True)
    lo.append(False)
    lo.append(False)
    var mid = BoolBuilder(3)
    mid.append(False)
    mid.append(True)
    mid.append(False)
    var b = record_batch(
        [
            lo.finish().to_dyn(),
            mid.finish().to_dyn(),
            array([10, 20, 30], int64).copy(),
        ],
        names=["lo", "mid", "v"],
    )
    var out = (
        DynRelation(InMemoryTable(b^))
        .project(
            ["bucket"],
            [
                DynValue(
                    case_when(
                        [column("lo"), column("mid")],
                        [
                            literal(DynScalar(Int64Scalar(1))),
                            literal(DynScalar(Int64Scalar(2))),
                        ],
                        literal(DynScalar(Int64Scalar(9))),
                    )
                )
            ],
        )
        .execute()
    )
    assert_true(out.columns[0].as_int64() == array([1, 2, 9], int64))
