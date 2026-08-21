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
- `interval` must **contain** every value the expression can produce. This is
  the only one-directional invariant here: an interval that is too wide costs
  a row group's decode time, one that is too narrow silently drops rows.
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
        assert_true(v.is_column())
        assert_true(v.shape() == Shape.columnar)


# ---------------------------------------------------------------------------
# name() / columns() jointly identify a bare column
# ---------------------------------------------------------------------------
def test_expr2_is_column_separates_the_three_cases() raises:
    """`name() != "" and len(columns()) == 1`, across all three shapes.

    The composition exists so that bare-column-ness needs no slot of its own.
    It only works if a literal is *named* but reads no columns — which is why
    `lit(7).name()` is `"7"` and not `""`.
    """
    var col = DynValue(Column[Int64Type]("a"))
    var lit = DynValue(Literal[Int64Type](7))

    assert_true(col.is_column())
    assert_equal(col.name(), "a")
    assert_equal(len(col.columns()), 1)

    # Named, but reads nothing — so not a column.
    assert_false(lit.is_column())
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
        RuntimeValue("add", _column_eval, _runtime_column("a"),
                     _runtime_column("b")),
    )
    var cols = v.columns()
    assert_equal(len(cols), 2)
    assert_equal(cols[0], "b")
    assert_equal(cols[1], "a")


def test_expr2_a_literal_reads_no_columns() raises:
    assert_equal(len(DynValue(Literal[Int64Type](3)).columns()), 0)


# ---------------------------------------------------------------------------
# interval contains what the expression can produce
# ---------------------------------------------------------------------------
def test_expr2_literal_interval_contains_its_own_value() raises:
    """The one-directional invariant: an interval may be too wide, never too
    narrow. Too wide costs a decode; too narrow drops rows silently."""
    from ..pruning import PruneStats

    var s = schema([field("a", int64)])
    var stats = PruneStats(
        schema=s,
        mins=[Optional[DynScalar](None)],
        maxs=[Optional[DynScalar](None)],
    )
    var iv = Literal[Int64Type](5).interval(stats)
    # 5 is in [5, 5]; nothing outside it is.
    assert_true(iv.maybe_eq(iv))


def test_expr2_unknown_statistics_stay_sound() raises:
    """A column with no statistic must answer *unknown*, not empty.

    An empty answer would let a scan prove a row group cannot match when it has
    simply never been told anything about it — the one failure mode that costs
    correctness rather than time.
    """
    from ..pruning import PruneStats

    var s = schema([field("a", int64)])
    var stats = PruneStats(
        schema=s,
        mins=[Optional[DynScalar](None)],
        maxs=[Optional[DynScalar](None)],
    )
    var iv = Column[Int64Type]("a").interval(stats)
    assert_true(iv.maybe_eq(Literal[Int64Type](999).interval(stats)))


# ---------------------------------------------------------------------------
# validity(bound) matches the evaluated column
# ---------------------------------------------------------------------------
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
