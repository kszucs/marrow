"""Plan nodes and the processors they become.

`logical.mojo` and `physical.mojo` are covered together because neither is
observable alone: a `Relation` is a description, so the only way to ask whether
it described the right thing is to run the `Processor` it builds. Each test
therefore states a claim about the plan and checks it against the rows.

The recurring failure these guard is a **schema that disagrees with the data** —
a `Project` whose declared fields do not match the columns it emits, or an empty
result whose schema names fields it has no columns for. Both run fine and
corrupt whatever reads them by index.
"""

from std.testing import assert_equal, assert_true

from ...builders import array
from ...dtypes import DynType, Int64Type, float64, int64
from ...execution import ExecContext
from ...tabular import RecordBatch, record_batch
from ..core import DynValue
from ..`comptime`.leaves import Column, Literal
from ..`comptime`.numeric import Add, Gt
from ..logical import DynRelation, Filter, InMemoryTable, Project


def _batch() raises -> RecordBatch:
    """`a` has a null so validity has something to report."""
    return record_batch(
        [
            array([1, 2, None, 4], int64).copy(),
            array([10, 20, 30, 40], int64).copy(),
        ],
        names=["a", "b"],
    )


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------
def test_a_dynamic_plan_runs_a_fused_predicate() raises:
    """The configuration the whole two-lane design exists to allow.

    The plan is erased and composed at run time; the predicate is a comptime
    type fused into one loop. That combination is what measures 1.46 MB against
    4.91 MB for the same plan with runtime expressions — and it only works
    because `DynValue` lets a `Filter` hold either lane without knowing which.
    """
    var plan = DynRelation(
        Filter(
            DynRelation(InMemoryTable(_batch())),
            DynValue(Gt(Column[Int64Type]("a"), Literal[Int64Type](2))),
        )
    )
    var out = plan.execute()

    # a = [1, 2, None, 4] -> only 4 > 2
    assert_equal(out.num_rows(), 1)
    ref a = out.columns[0].as_int64()
    assert_equal(a[0].value(), 4)


def test_a_null_predicate_does_not_select() raises:
    """SQL's rule, and the reason `Filter` must not read the data bit alone.

    `None > 2` is NULL, not false — but the SIMD lane still produced a bit for
    that row. If the filter selected on data bits, the null row's payload would
    decide whether it survives.
    """
    var plan = DynRelation(
        Filter(
            DynRelation(InMemoryTable(_batch())),
            # every non-null row passes, so only the null's treatment shows
            DynValue(Gt(Column[Int64Type]("a"), Literal[Int64Type](-100))),
        )
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 3)  # 1, 2, 4 — the null is not selected


def test_filter_preserves_its_input_schema() raises:
    """A filter changes which rows survive, never which columns exist."""
    var b = _batch()
    var plan = DynRelation(
        Filter(
            DynRelation(InMemoryTable(b.copy())),
            DynValue(Gt(Column[Int64Type]("a"), Literal[Int64Type](2))),
        )
    )
    assert_true(plan.schema() == b.schema)
    assert_equal(plan.execute().num_columns(), b.num_columns())


def test_an_empty_result_is_a_well_formed_batch() raises:
    """Zero rows still means one zero-length column per field.

    A schema naming fields beside an empty column list leaves `num_columns()`
    at 0, so anything walking columns by schema index runs off the end — and
    exporting it over the C Data interface returns NULL without setting an
    exception.
    """
    var b = _batch()
    var plan = DynRelation(
        Filter(
            DynRelation(InMemoryTable(b.copy())),
            DynValue(Gt(Column[Int64Type]("a"), Literal[Int64Type](999))),
        )
    )
    var out = plan.execute()
    assert_equal(out.num_rows(), 0)
    assert_equal(out.num_columns(), len(b.schema.fields))


# ---------------------------------------------------------------------------
# Project — the caller that justifies dtype() and name()


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------
def test_project_carries_a_bare_column_field_whole() raises:
    """A projected pass-through keeps its source `Field`, not just its dtype.

    Rebuilding the field from `dtype()` alone loses `nullable`, so projecting
    a column would produce a *different* schema for it than selecting the same
    column does. `expr/` records that divergence with `nullable` False
    becoming True.
    """
    var b = record_batch([array([1, 2], int64).copy()], names=["a"])
    # `a` is non-nullable here; the projection must not widen it.
    var src = b.schema.fields[0].nullable

    var p = Project(
        DynRelation(InMemoryTable(b.copy())),
        ["out"],
        [DynValue(Column[Int64Type]("a"))],
    )
    assert_equal(p.schema().fields[0].nullable, src)
    assert_true(p.schema().fields[0].dtype == DynType(int64))


def test_project_names_a_computed_column_from_its_dtype() raises:
    """A computed value has no `Field` to carry, so `dtype()` answers instead.

    This is `dtype()`'s reason to exist: the schema must be known *before*
    anything runs, and `expr/` got it by evaluating against a zero-row batch.
    """
    var b = _batch()
    var p = Project(
        DynRelation(InMemoryTable(b.copy())),
        ["sum"],
        [DynValue(Add(Column[Int64Type]("a"), Column[Int64Type]("b")))],
    )
    assert_equal(p.schema().fields[0].name, "sum")
    assert_true(p.schema().fields[0].dtype == DynType(int64))


def test_project_schema_matches_what_it_produces() raises:
    """The soundness property: the declared schema and the executed batch agree.

    A schema computed statically can disagree with the batch; one derived by
    evaluating cannot. Trading the probe for `dtype()` is what makes this
    worth asserting.
    """
    var b = _batch()
    var plan = DynRelation(
        Project(
            DynRelation(InMemoryTable(b.copy())),
            ["sum", "orig"],
            [
                DynValue(Add(Column[Int64Type]("a"), Column[Int64Type]("b"))),
                DynValue(Column[Int64Type]("a")),
            ],
        )
    )
    var out = plan.execute()
    assert_true(out.schema == plan.schema())
    assert_equal(out.num_columns(), 2)
    assert_equal(out.num_rows(), b.num_rows())


def test_project_rejects_mismatched_names_and_values() raises:
    """Two parallel lists that must agree, checked where they are supplied."""
    var raised = False
    try:
        _ = Project(
            DynRelation(InMemoryTable(_batch())),
            ["only_one"],
            [
                DynValue(Column[Int64Type]("a")),
                DynValue(Column[Int64Type]("b")),
            ],
        )
    except e:
        raised = True
        assert_true("project" in String(e))
    assert_true(raised)


# ---------------------------------------------------------------------------
# builders — `col` and `lit` select a lane by what the caller knows
