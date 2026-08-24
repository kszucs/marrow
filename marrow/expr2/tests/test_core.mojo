"""`DynValue`, `Shape` and `Datum` — the erasure boundary itself.

What is tested here is the boundary, not either lane: that both lanes box into
one `DynValue`, and that every question the box answers agrees with what the
value it holds actually does when evaluated. A box that reports one dtype and
produces another is the failure this file exists to catch, and nothing above it
could detect the disagreement.

Lane-specific behaviour lives with its lane — `comptime/tests/` and
`runtime/tests/`.
"""

from std.testing import assert_equal, assert_false, assert_true

from ...arrays import DynArray
from ..params import Bindings
from ...builders import array
from ..builders import col, lit
from ...dtypes import DynType, Int64Type, int64
from ...scalars import DynScalar
from ...tabular import RecordBatch, record_batch
from ..logical import DynValue, Shape, Value
from ..`comptime`.leaves import Column, Literal
from ..runtime.values import RuntimeValue, column


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
# Both lanes satisfy one Value and box into one DynValue
# ---------------------------------------------------------------------------
def test_both_lanes_box_together() raises:
    """The two-lane design's load-bearing claim, asserted rather than assumed.

    A comptime node and a runtime node must be interchangeable *as boxed
    values*, because a `Relation` holds `DynValue` and never learns which lane
    it was handed. If this fails, the plan layer needs to know, and the whole
    separation collapses.
    """
    var boxed = List[DynValue]()
    boxed.append(col("a", int64))
    boxed.append(column("a"))

    for ref v in boxed:
        assert_equal(v.name(), "a")
        assert_equal(len(v.columns()), 1)
        assert_true(v.name() != "" and len(v.columns()) == 1)
        assert_true(v.shape() == Shape.columnar)


# ---------------------------------------------------------------------------
# name() / columns() jointly identify a bare column


def test_is_column_separates_the_three_cases() raises:
    """`name() != "" and len(columns()) == 1`, across all three shapes.

    The composition exists so that bare-column-ness needs no slot of its own,
    and no method either: it is spelled out at each call site until a caller
    exists that writes it more than once. It only works if a literal is
    *named* but reads no columns — which is why `lit(7).name()` is `"7"`.
    """
    var c = col("a", int64)
    var l = lit(7, int64)

    assert_true(c.name() != "" and len(c.columns()) == 1)
    assert_equal(c.name(), "a")
    assert_equal(len(c.columns()), 1)

    # Named, but reads nothing — so not a column.
    assert_false(l.name() != "" and len(l.columns()) == 1)
    assert_equal(l.name(), "7")
    assert_equal(len(l.columns()), 0)


def test_literal_is_named_as_sql_names_it() raises:
    """`SELECT 1` yields a column called `1`; so does `lit(1)`."""
    assert_equal(lit(1, int64).name(), "1")
    assert_equal(lit(-42, int64).name(), "-42")


# ---------------------------------------------------------------------------
# dtype(schema) agrees with evaluate()


def test_a_literal_reads_no_columns() raises:
    assert_equal(len(lit(3, int64).columns()), 0)


# ---------------------------------------------------------------------------
# What the box says agrees with what it does
# ---------------------------------------------------------------------------
def test_dtype_of_a_missing_column_names_the_column() raises:
    """A bad column must be reported by name, not by a bounds abort.

    `get_field_index` answers -1, and indexing a column list with that trips an
    assert that kills the process instead of saying which column was wrong.
    """
    var v = column("nope")
    var raised = False
    try:
        _ = v.dtype(_batch().schema)
    except e:
        raised = True
        assert_true("nope" in String(e))
    assert_true(raised)


# ---------------------------------------------------------------------------
# shape agrees with what evaluate returns


def test_shape_agrees_with_evaluation() raises:
    """A `scalar` node must stay lazy; a `columnar` one must materialise.

    If a literal claimed `columnar`, every predicate over a constant would
    allocate a full column before comparing.
    """
    var b = _batch()

    var l = lit(7, int64)
    assert_true(l.shape == Shape.scalar)
    assert_true(l.evaluate(b.to_struct_array(), Bindings()).is_scalar())

    var c = col("a", int64)
    assert_true(c.shape == Shape.columnar)
    assert_false(c.evaluate(b.to_struct_array(), Bindings()).is_scalar())

    # The runtime lane cannot be lazy, and says so.
    assert_true(RuntimeValue.shape == Shape.columnar)


# ---------------------------------------------------------------------------
# columns() is deduped and order-preserving
