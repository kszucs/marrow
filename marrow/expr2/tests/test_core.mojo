"""Soundness of the `expr2` spine.

These do not test that a kernel computes the right numbers — that is the golden
corpus's job, against DuckDB. They test the **invariants the design rests on**,
each of which is a claim one method makes about another. A design where two
methods can disagree is one where a caller must know which to trust, and every
such pair here has already cost something in `expr/`:

- `dtype(schema)` must agree with what `evaluate` actually produces. `expr/`
  had no `dtype` and computed output types by evaluating against a zero-row
  batch, which cannot disagree because it *is* the evaluation. Answering
  statically reintroduces the possibility, so it gets a test.
- `shape` must agree with what `evaluate` returns — scalar or array. A node
  claiming `scalar` while materialising a column would make `into_array`
  broadcast something already broadcast.
- `name()` and `columns()` must jointly identify a bare column, because four
  callers ask exactly that composition and nothing else answers it.
- `validity(bound)` must match the null pattern of the evaluated column.

Both lanes are asserted through the *same* box wherever the property is about
the erased surface, because "both lanes agree" is the property the two-lane
design is staking everything on.
"""

from std.testing import assert_equal, assert_false, assert_true

from ...arrays import DynArray
from ...builders import array
from ...dtypes import DynType, Int64Type, field, int64
from ...scalars import DynScalar
from ...schema import schema
from ...tabular import RecordBatch, record_batch
from ..core import Datum, DynValue, Shape, into_array
from ..`comptime`.leaves import Column, Literal
from ..`comptime`.numeric import Add, Gt
from ..plan import DynRelation, Filter, InMemoryTable, Project
from ..runtime.values import Payload, RuntimeValue


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _batch() raises -> RecordBatch:
    """`a` has a null so validity has something to report."""
    return record_batch(
        [
            array([1, 2, None, 4], int64).copy(),
            array([10, 20, 30, 40], int64).copy(),
        ],
        names=["a", "b"],
    )


def _column_eval(
    kids: List[DynArray], p: Payload, b: RecordBatch
) raises -> DynArray:
    return b.column(p[String]).copy()


def _runtime_column(var name: String) -> RuntimeValue:
    return RuntimeValue("column", _column_eval, Payload(name^))


def _runtime_literal(var v: DynScalar) -> RuntimeValue:
    return RuntimeValue("literal", _column_eval, Payload(v^))


# ---------------------------------------------------------------------------
# Both lanes satisfy one Value and box into one DynValue
# ---------------------------------------------------------------------------
def test_expr2_both_lanes_box_together() raises:
    """The two-lane design's load-bearing claim, asserted rather than assumed.

    A comptime node and a runtime node must be interchangeable *as boxed
    values*, because a `Relation` holds `DynValue` and never learns which lane
    it was handed. If this fails, the plan layer needs to know, and the whole
    separation collapses.
    """
    var boxed = List[DynValue]()
    boxed.append(DynValue(Column[Int64Type]("a")))
    boxed.append(DynValue(_runtime_column("a")))

    for ref v in boxed:
        assert_equal(v.name(), "a")
        assert_equal(len(v.columns()), 1)
        assert_true(v.name() != "" and len(v.columns()) == 1)
        assert_true(v.shape() == Shape.columnar)


# ---------------------------------------------------------------------------
# name() / columns() jointly identify a bare column
# ---------------------------------------------------------------------------
def test_expr2_is_column_separates_the_three_cases() raises:
    """`name() != "" and len(columns()) == 1`, across all three shapes.

    The composition exists so that bare-column-ness needs no slot of its own,
    and no method either: it is spelled out at each call site until a caller
    exists that writes it more than once. It only works if a literal is
    *named* but reads no columns — which is why `lit(7).name()` is `"7"`.
    """
    var col = DynValue(Column[Int64Type]("a"))
    var lit = DynValue(Literal[Int64Type](7))

    assert_true(col.name() != "" and len(col.columns()) == 1)
    assert_equal(col.name(), "a")
    assert_equal(len(col.columns()), 1)

    # Named, but reads nothing — so not a column.
    assert_false(lit.name() != "" and len(lit.columns()) == 1)
    assert_equal(lit.name(), "7")
    assert_equal(len(lit.columns()), 0)


def test_expr2_literal_is_named_as_sql_names_it() raises:
    """`SELECT 1` yields a column called `1`; so does `lit(1)`."""
    assert_equal(DynValue(Literal[Int64Type](1)).name(), "1")
    assert_equal(DynValue(Literal[Int64Type](-42)).name(), "-42")


# ---------------------------------------------------------------------------
# dtype(schema) agrees with evaluate()
# ---------------------------------------------------------------------------
def test_expr2_dtype_agrees_with_evaluation_comptime() raises:
    """The comptime lane answers from `Type` without touching the batch.

    That is the whole reason `dtype` exists rather than probing — so the two
    answers being equal is the property, not an implementation detail.
    """
    var b = _batch()
    var v = DynValue(Column[Int64Type]("a"))
    var produced = into_array(v.evaluate(b), b.num_rows()).dtype()
    assert_true(v.dtype(b.schema) == produced)
    assert_true(v.dtype(b.schema) == DynType(int64))


def test_expr2_dtype_agrees_with_evaluation_runtime() raises:
    """Same claim for the lane that has to look itself up in the schema."""
    var b = _batch()
    var v = DynValue(_runtime_column("b"))
    var produced = into_array(v.evaluate(b), b.num_rows()).dtype()
    assert_true(v.dtype(b.schema) == produced)


def test_expr2_dtype_of_a_missing_column_names_the_column() raises:
    """A bad column must be reported by name, not by a bounds abort.

    `get_field_index` answers -1, and indexing a column list with that trips an
    assert that kills the process instead of saying which column was wrong.
    """
    var v = DynValue(_runtime_column("nope"))
    var raised = False
    try:
        _ = v.dtype(_batch().schema)
    except e:
        raised = True
        assert_true("nope" in String(e))
    assert_true(raised)


# ---------------------------------------------------------------------------
# shape agrees with what evaluate returns
# ---------------------------------------------------------------------------
def test_expr2_shape_agrees_with_evaluation() raises:
    """A `scalar` node must stay lazy; a `columnar` one must materialise.

    If a literal claimed `columnar`, every predicate over a constant would
    allocate a full column before comparing.
    """
    var b = _batch()

    var lit = Literal[Int64Type](7)
    assert_true(lit.shape == Shape.scalar)
    assert_true(lit.evaluate(b).isa[DynScalar]())

    var col = Column[Int64Type]("a")
    assert_true(col.shape == Shape.columnar)
    assert_true(col.evaluate(b).isa[DynArray]())

    # The runtime lane cannot be lazy, and says so.
    assert_true(RuntimeValue.shape == Shape.columnar)


# ---------------------------------------------------------------------------
# columns() is deduped and order-preserving
# ---------------------------------------------------------------------------
def test_expr2_columns_are_deduped_in_first_seen_order() raises:
    """Projection pushdown reads this list; a repeat would narrow twice and a
    reorder would build a schema in the wrong order."""
    var v = RuntimeValue(
        "add",
        _column_eval,
        _runtime_column("b"),
        RuntimeValue(
            "add", _column_eval, _runtime_column("a"), _runtime_column("b")
        ),
    )
    var cols = v.columns()
    assert_equal(len(cols), 2)
    assert_equal(cols[0], "b")
    assert_equal(cols[1], "a")


def test_expr2_a_literal_reads_no_columns() raises:
    assert_equal(len(DynValue(Literal[Int64Type](3)).columns()), 0)


def test_expr2_validity_matches_the_bound_column() raises:
    """`lane` produces data bits only, so validity is a separate contract.

    Reading it from the `Bound` rather than the batch is what stops the second
    pass `expr/` needed; the property is that it still reports the same nulls.
    """
    var b = _batch()
    var c = Column[Int64Type]("a")
    var bound = c.bind(b)
    var v = c.validity(bound)

    assert_true(v)
    # `a` is [1, 2, None, 4] — exactly one null, at index 2.
    assert_false(v.value()[2])
    assert_true(v.value()[0])


def test_expr2_a_literal_is_never_null() raises:
    var b = _batch()
    var l = Literal[Int64Type](7)
    assert_false(Bool(l.validity(l.bind(b))))


# ---------------------------------------------------------------------------
# Fusion: the lane computes data bits, validity records what they mean
# ---------------------------------------------------------------------------
def test_expr2_a_comparison_over_a_null_is_null_not_false() raises:
    """The defect class this whole validity contract exists to prevent.

    A SIMD lane compares whatever is in the slot, so the *data* bit for a null
    row is whatever the payload happened to be — usually zero, which reads as
    `False`. Only the validity bitmap records that the bit is meaningless.
    Reading the data bit without it is exactly what made NULL join keys match
    each other in `expr/`.
    """
    var b = _batch()  # a = [1, 2, None, 4]
    var pred = Gt(Column[Int64Type]("a"), Literal[Int64Type](2))
    var arr = into_array(pred.evaluate(b), b.num_rows())
    ref out = arr.as_bool()

    assert_equal(out.null_count(), 1)
    assert_true(out.is_null(2))
    # The rows that are not null answer the comparison.
    assert_false(out[0].value())  # 1 > 2
    assert_false(out[1].value())  # 2 > 2
    assert_true(out[3].value())  # 4 > 2


def test_expr2_arithmetic_propagates_nulls_through_fusion() raises:
    """Null in, null out — across a fused subtree, not just one node."""
    var b = _batch()
    var sum = Add(Column[Int64Type]("a"), Column[Int64Type]("b"))
    var arr = into_array(sum.evaluate(b), b.num_rows())
    ref out = arr.as_int64()

    assert_equal(out.null_count(), 1)
    assert_true(out.is_null(2))
    assert_equal(out[0].value(), 11)
    assert_equal(out[3].value(), 44)


def test_expr2_a_fused_subtree_is_one_expression() raises:
    """`(a + b) > 10` fuses three nodes; the null still propagates to the top.

    The point is not the arithmetic — it is that `bind` descends the whole
    subtree once and validity survives two levels, which is what a single
    fused pass has to preserve.
    """
    var b = _batch()
    var pred = Gt(
        Add(Column[Int64Type]("a"), Column[Int64Type]("b")),
        Literal[Int64Type](10),
    )
    var arr = into_array(pred.evaluate(b), b.num_rows())
    ref out = arr.as_bool()

    assert_equal(out.null_count(), 1)
    assert_true(out.is_null(2))
    assert_true(out[0].value())  # 1 + 10 = 11 > 10
    assert_true(out[3].value())  # 4 + 40 = 44 > 10


def test_expr2_a_literal_only_expression_stays_scalar() raises:
    """`Shape.scalar` must survive composition, or every constant folds into a
    column before it is used."""
    var b = _batch()
    var both = Add(Literal[Int64Type](1), Literal[Int64Type](2))
    assert_true(both.shape == Shape.scalar)
    assert_true(both.evaluate(b).isa[DynScalar]())


# ---------------------------------------------------------------------------
# End to end: a dynamic plan holding a fused predicate
# ---------------------------------------------------------------------------
def test_expr2_a_dynamic_plan_runs_a_fused_predicate() raises:
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


def test_expr2_a_null_predicate_does_not_select() raises:
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


def test_expr2_filter_preserves_its_input_schema() raises:
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


def test_expr2_an_empty_result_is_a_well_formed_batch() raises:
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
def test_expr2_project_carries_a_bare_column_field_whole() raises:
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


def test_expr2_project_names_a_computed_column_from_its_dtype() raises:
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


def test_expr2_project_schema_matches_what_it_produces() raises:
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


def test_expr2_project_rejects_mismatched_names_and_values() raises:
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
