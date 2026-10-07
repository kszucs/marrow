# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""What a plan expects to produce, and what it says when it does not know.

Three kinds of case, and the second is the one that matters.

**Plans with known answers.** A four-row batch produces four rows, a `LIMIT 2`
produces two, an aggregate with no keys produces one, and a join of known
cardinalities produces the containment estimate — each asserted as an exact
figure so that a formula quietly drifting is a failure rather than a different
shade of plausible.

**Unknown stays unknown.** A source nobody read a footer for, a computed
column, a string column's width: each must reach the root as unknown rather
than as a fabricated number, because a cost model spends a wrong figure exactly
as confidently as a right one. Several cases therefore assert `is_known()` is
false, which is the assertion a fabricating estimator fails and a merely
imprecise one passes.

**Provable emptiness is exact.** `Filter` answers exactly zero when the
predicate cannot match the child's bounds, and it reaches that answer through
`Value.mask` — the same walk `Index.read_plan` runs over a footer. The cases
below pin both halves: the proof when the bounds decide it, and
`DEFAULT_SELECTIVITY` when they do not, which must never itself be zero.
"""

from std.math import nan
from std.python import Python
from std.sys import stderr
from std.testing import assert_equal, assert_false, assert_true
from std.os.path import join

from ...utils.testing import ScratchDir
from ...builders import array
from ...dtypes import (
    Field,
    FixedSizeBinaryType,
    binary,
    date32,
    dictionary,
    field,
    fixed_size_list_,
    float64,
    int16,
    int32,
    int64,
    list_,
    null,
    second,
    string,
    struct_,
    timestamp,
    uint64,
)
from ...kernels.join import (
    BUILD_LEFT,
    JOIN_ANY,
    BUILD_RIGHT,
    JOIN_ANTI,
    JOIN_FULL,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_RIGHT_ANTI,
    JOIN_RIGHT_SEMI,
    JOIN_SEMI,
    JoinKind,
)
from ...io import FileSink
from ...parquet.codecs import Compression
from ...parquet.reader import PageBounds, ParquetFile
from ...parquet.writer import FileWriter
from ...scalars import (
    Date32Scalar,
    DynScalar,
    Float64Scalar,
    Int32Scalar,
    Int64Scalar,
    StringScalar,
    TimestampScalar,
    UInt64Scalar,
)
from ...schema import Schema, schema
from ...tabular import RecordBatch, Table, record_batch
from ..builders import col, lit, param, scan, table
from ..analyze import analyze, summarize
from ..estimates import (
    Approx,
    ColumnEstimate,
    Cost,
    DEFAULT_BINARY_WIDTH,
    DEFAULT_SELECTIVITY,
    DEFAULT_STRING_WIDTH,
    Estimate,
    Size,
)
from ..index import ColumnZones, Index
from ..logical import (
    DynRelation,
    DynValue,
    EmptyRelation,
    InMemoryTable,
    JoinChain,
    ParquetScan,
    ScanPath,
)
from ..optimizer import AllRules, NoRules


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _batch() raises -> RecordBatch:
    """Four rows, two `int64` columns, one null in `a`."""
    return record_batch(
        [
            array([1, 2, None, 4], int64).copy(),
            array([10, 20, 30, 40], int64).copy(),
        ],
        names=["a", "b"],
    )


def _column(
    var name: String,
    lo: Int,
    hi: Int,
    ndv: Approx = Approx(),
) raises -> ColumnEstimate:
    """One `int64` column summarised by hand, so a formula is tested against
    figures the case chose rather than against whatever a file happened to
    hold."""
    return ColumnEstimate(
        name^,
        Int64Scalar(Scalar[int64.native](lo)).to_dyn(),
        Int64Scalar(Scalar[int64.native](hi)).to_dyn(),
        nulls=Approx.exact(0),
        ndv=ndv,
        width=Approx.exact(8),
    )


def _source(
    var columns: List[ColumnEstimate], rows: Approx
) raises -> DynRelation:
    """A scan told what its footer says, over a file nothing opens.

    No case here executes, so the path never has to exist: the whole point of
    attaching statistics rather than reading them is that estimating a plan
    touches no I/O.
    """
    var fields = List[Field](capacity=len(columns))
    for ref c in columns:
        fields.append(field(c.name.copy(), int64))
    var node = ParquetScan(
        ScanPath(String("/nonexistent.parquet")), schema(fields^)
    ).with_statistics(Estimate(rows, columns^))
    var out: DynRelation = node^
    return out^


def _ndv_of(var distinct_counts: List[Int]) raises -> Approx:
    """The `ndv` `Estimate.from_index` reduces a single column's per-chunk
    distinct counts to, with the bounds left unrecorded so only the reduction
    is under test."""
    var n = len(distinct_counts)
    var mins = List[DynScalar](capacity=n)
    var maxes = List[DynScalar](capacity=n)
    var nulls = List[Int](capacity=n)
    for _ in range(n):
        mins.append(Int64Scalar(Optional[Scalar[int64.native]](None)).to_dyn())
        maxes.append(Int64Scalar(Optional[Scalar[int64.native]](None)).to_dyn())
        nulls.append(-1)
    var index = Index(
        List[Int](length=n, fill=-1),
        [ColumnZones(String("a"), mins^, maxes^, nulls^, distinct_counts^)],
    )
    return (
        Estimate.from_index(index, schema([field("a", int64)])).columns[0].ndv
    )


def _bare_scan() raises -> DynRelation:
    """A scan nobody read a footer for — the unknown source."""
    var node = ParquetScan(
        ScanPath(String("/nonexistent.parquet")),
        schema([field("a", int64), field("b", int64)]),
    )
    var out: DynRelation = node^
    return out^


# ---------------------------------------------------------------------------
# Approx -- the carrier
# ---------------------------------------------------------------------------
def test_estimate_unknown_absorbs_every_operator() raises:
    """The property the whole design rests on: nothing turns an unknown into a
    number, not even an operation whose other operand is exact."""
    var known = Approx.exact(10)
    var nothing = Approx.unknown()

    assert_false((known + nothing).is_known())
    assert_false((known - nothing).is_known())
    assert_false((known * nothing).is_known())
    assert_false((known // nothing).is_known())
    assert_false(known.at_most(nothing).is_known())
    assert_false(known.at_least(nothing).is_known())
    assert_false(nothing.scaled(50).is_known())
    assert_false(nothing.times(3).is_known())
    assert_false(nothing.to_estimate().is_known())


def test_estimate_exactness_never_improves() raises:
    """An estimate mixed with an exact figure yields an estimate; nothing
    here promotes one back."""
    assert_true((Approx.exact(2) + Approx.exact(3)).is_exact())
    assert_false((Approx.exact(2) + Approx.estimated(3)).is_exact())
    assert_equal(String(Approx.exact(2) + Approx.estimated(3)), "~5")
    assert_false(Approx.exact(2).to_estimate().is_exact())


def test_estimate_division_needs_a_known_positive_divisor() raises:
    assert_true((Approx.exact(10) // Approx.exact(2)) == Approx.exact(5))
    assert_false((Approx.exact(10) // Approx.estimated(2)).is_exact())
    assert_false((Approx.exact(10) // Approx.exact(0)).is_known())


def test_estimate_scaling_never_reaches_zero() raises:
    """Zero is reserved for *provably* empty.

    20% of four rows rounds to one, not to zero: a filter that might match and
    one proven to match nothing are read identically downstream, and only the
    second is a proof.
    """
    assert_equal(Approx.exact(4).scaled(DEFAULT_SELECTIVITY).known().value(), 1)
    assert_equal(
        Approx.exact(100).scaled(DEFAULT_SELECTIVITY).known().value(), 20
    )
    # a genuinely empty input stays empty
    assert_equal(Approx.exact(0).scaled(DEFAULT_SELECTIVITY).known().value(), 0)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def test_estimate_in_memory_table_counts_rows_and_nulls_exactly() raises:
    """Both are stored on the batch, so both are read rather than guessed —
    and nothing else is, because finding a minimum means a scan."""
    var est = table(_batch()).estimate()

    assert_true(est.rows.is_exact())
    assert_equal(est.rows.known().value(), 4)
    assert_equal(est.columns[0].nulls.known().value(), 1)
    assert_equal(est.columns[1].nulls.known().value(), 0)
    assert_false(est.columns[0].ndv.is_known())
    assert_false(est.columns[0].min.is_valid())
    assert_equal(est.row_width().known().value(), 16)


def test_estimate_a_source_without_a_footer_knows_nothing() raises:
    """And says so all the way to the root, rather than defaulting to a
    figure the plan above would then treat as fact."""
    var plan = _bare_scan().filter(col("a", int64) > lit(2, int64))
    var est = plan.estimate()

    assert_false(est.rows.is_known())
    assert_false(plan.cost().total().is_known())
    # the shape still lines up with the schema, so a node above can index it
    assert_equal(len(est.columns), 2)


def test_estimate_a_variable_width_column_is_priced_at_a_default() raises:
    """A string's average length is a fact about the data, and nothing here
    has read any: it is priced at `DEFAULT_STRING_WIDTH`, an estimate, and so
    is the row it is part of."""
    var batch = record_batch(
        [array([1, 2], int64).copy(), array(["x", "yy"]).copy()],
        names=["a", "s"],
    )
    var est = table(batch^).estimate()

    assert_equal(est.columns[0].width.known().value(), 8)
    assert_equal(est.columns[1].width.known().value(), DEFAULT_STRING_WIDTH)
    assert_false(est.columns[1].width.is_exact())
    assert_equal(est.row_width().known().value(), 8 + DEFAULT_STRING_WIDTH)
    assert_false(est.row_width().is_exact())


def test_estimate_empty_relation_is_exactly_nothing() raises:
    """The one cardinality in the system that is a certainty.

    Asserted on both spellings of it — the `LIMIT 0` and the `EmptyRelation`
    the rewrite replaces it with — because the two must agree or a rule would
    change a plan's estimated cardinality while preserving its answer.
    """
    var plan = table(_batch()).limit(0)
    assert_true(plan.estimate().rows.is_exact())
    assert_equal(plan.estimate().rows.known().value(), 0)

    var optimized = plan.optimize[AllRules]()
    assert_true(optimized.isa[EmptyRelation]())
    assert_equal(optimized.estimate().rows.known().value(), 0)


# ---------------------------------------------------------------------------
# Filter -- selectivity is `Value.mask` over one chunk
# ---------------------------------------------------------------------------
def test_estimate_filter_proves_emptiness_from_the_bounds() raises:
    """`a` never exceeds 9, so `a > 100` keeps nothing — and the answer is
    **exact**, because it rests on the bound being sound rather than on the
    row count being right.

    This is the same computation `Index.read_plan` runs over a footer, reached
    through the same `Value.mask`; a one-chunk index is the only difference.
    """
    var src = _source([_column(String("a"), 0, 9)], Approx.exact(100))
    var est = src.filter(col("a", int64) > lit(100, int64)).estimate()

    assert_true(est.rows.is_exact())
    assert_equal(est.rows.known().value(), 0)


def test_estimate_filter_keeps_its_share_when_the_bounds_decide_nothing() raises:
    """`a > 5` against `[0, 9]` cannot be decided either way: it keeps four of
    the ten values a histogram of `a` holds, and is marked as an estimate."""
    var src = _source([_column(String("a"), 0, 9)], Approx.exact(100))
    var est = src.filter(col("a", int64) > lit(5, int64)).estimate()

    assert_false(est.rows.is_exact())
    assert_true(est.rows.is_known())
    assert_equal(est.rows.known().value(), 40)


def test_estimate_filter_keeps_bounds_and_counts() raises:
    """A filter removes values, so `[min, max]` still contains what is left,
    and a column proven free of NULLs stays so."""
    var src = _source([_column(String("a"), 0, 9)], Approx.exact(100))
    var est = src.filter(col("a", int64) > lit(5, int64)).estimate()

    assert_true(est.columns[0].min.is_valid())
    assert_equal(est.columns[0].min.as_int64().value(), 0)
    assert_equal(est.columns[0].max.as_int64().value(), 9)
    assert_true(est.columns[0].nulls == Approx.exact(0))


def test_estimate_filter_on_an_unknown_source_stays_unknown() raises:
    """A percentage of nothing is nothing known — not zero, and not the
    percentage."""
    var est = _bare_scan().filter(col("a", int64) > lit(5, int64)).estimate()
    assert_false(est.rows.is_known())


# ---------------------------------------------------------------------------
# Project, Limit, Sort, Window
# ---------------------------------------------------------------------------
def test_estimate_project_carries_only_what_passes_through() raises:
    """`a` is emitted as itself and keeps its summary; `a + b` is a new column
    and keeps only the width its dtype implies."""
    var values: List[DynValue] = [
        col("a", int64),
        col("a", int64) + col("b", int64),
    ]
    var plan = table(_batch()).project(["a", "sum"], values^)
    var est = plan.estimate()

    assert_equal(est.rows.known().value(), 4)
    assert_equal(est.columns[0].nulls.known().value(), 1)
    assert_false(est.columns[1].nulls.is_known())
    assert_equal(est.columns[1].width.known().value(), 8)


def test_estimate_limit_is_the_one_exact_reduction() raises:
    """Arithmetic on a cardinality, not an assumption about data — so an exact
    input gives an exact output where a filter could only guess."""
    var plan = table(_batch()).limit(2)
    assert_true(plan.estimate().rows.is_exact())
    assert_equal(plan.estimate().rows.known().value(), 2)

    # a limit larger than the input does not invent rows
    assert_equal(table(_batch()).limit(99).estimate().rows.known().value(), 4)

    # and an offset comes off the front
    assert_equal(
        table(_batch()).limit(99, offset=3).estimate().rows.known().value(), 1
    )


def test_estimate_sort_moves_rows_without_changing_them() raises:
    """Every count and every bound survives an ordering exactly."""
    var plan = table(_batch()).sort_by([col("a", int64)], [True])
    var est = plan.estimate()

    assert_true(est.rows.is_exact())
    assert_equal(est.rows.known().value(), 4)
    assert_equal(est.columns[0].nulls.known().value(), 1)


def test_estimate_window_appends_columns_it_cannot_summarise() raises:
    """The rows and the input columns carry; the appended one does not."""
    var plan = table(_batch()).with_columns(
        ["r"], [col("a", int64).lag().over()]
    )
    var est = plan.estimate()

    assert_equal(est.rows.known().value(), 4)
    assert_equal(est.columns[0].nulls.known().value(), 1)
    assert_false(est.columns[2].nulls.is_known())


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def test_estimate_aggregate_without_keys_is_exactly_one_row() raises:
    """`SELECT sum(x) FROM t` produces one row whatever `t` holds, so this is
    not an estimate."""
    var plan = table(_batch()).aggregate(
        [col("a", int64).sum().alias("total")], List[DynValue]()
    )
    var est = plan.estimate()

    assert_true(est.rows.is_exact())
    assert_equal(est.rows.known().value(), 1)


def test_estimate_aggregate_is_capped_by_the_rows_it_read() raises:
    """No distinct count is recorded, so the key falls back to the row count
    and the product caps there: at most as many groups as rows."""
    var plan = table(_batch()).aggregate(
        [col("b", int64).sum().alias("total")], [col("a", int64)]
    )
    var est = plan.estimate()

    assert_false(est.rows.is_exact())
    assert_equal(est.rows.known().value(), 4)


def test_estimate_aggregate_uses_a_recorded_distinct_count() raises:
    """With an `ndv` in hand the group count is that, not the row count."""
    var src = _source(
        [
            _column(String("a"), 0, 9, ndv=Approx.exact(3)),
            _column(String("b"), 0, 99),
        ],
        Approx.exact(100),
    )
    var plan = src.aggregate(
        [col("b", int64).sum().alias("total")], [col("a", int64)]
    )
    var est = plan.estimate()

    assert_equal(est.rows.known().value(), 3)


def test_estimate_aggregate_keeps_key_bounds_off_its_aggregates() raises:
    """An aggregate may carry a key's name, and must not inherit its bounds.

    `sum(b) AS a` grouped by `a` emits two fields called `a`, so a name-keyed
    copy would give the sum the key's `[0, 9]` — and a predicate above would
    prune a group it had to read. Keys come first and aggregates after, which
    is the only thing that distinguishes them.
    """
    var src = _source(
        [
            _column(String("a"), 0, 9, ndv=Approx.exact(3)),
            _column(String("b"), 100, 200),
        ],
        Approx.exact(100),
    )
    var plan = src.aggregate(
        [col("b", int64).sum().alias("a")], [col("a", int64)]
    )
    var est = plan.estimate()

    assert_equal(len(est.columns), 2)
    assert_equal(est.columns[0].max.as_int64().value(), 9)
    assert_false(est.columns[1].max.is_valid())


def test_estimate_aggregate_on_a_computed_key_knows_nothing() raises:
    """`a + b` has no name and no summary, and `grouped` will not invent one
    for it."""
    var plan = table(_batch()).aggregate(
        [col("b", int64).sum().alias("total")],
        [col("a", int64) + col("b", int64)],
    )
    assert_false(plan.estimate().rows.is_known())


# ---------------------------------------------------------------------------
# Join
# ---------------------------------------------------------------------------
def _join_sides() raises -> Tuple[DynRelation, DynRelation]:
    """100 rows over 10 distinct keys, joined to 1000 rows over 50."""
    var left = _source(
        [_column(String("k"), 0, 9, ndv=Approx.exact(10))], Approx.exact(100)
    )
    var right = _source(
        [_column(String("k"), 0, 49, ndv=Approx.exact(50))], Approx.exact(1000)
    )
    return (left^, right^)


def test_estimate_join_divides_by_the_larger_key_domain() raises:
    """`100 * 1000 / max(10, 50)` — containment, so the smaller domain is
    assumed to sit inside the larger."""
    var sides = _join_sides()
    var plan = sides[0].join(sides[1].copy(), [0], [0], JOIN_INNER)
    var est = plan.estimate()

    assert_false(est.rows.is_exact())
    assert_equal(est.rows.known().value(), 2000)


def test_estimate_outer_joins_are_lower_bounded() raises:
    """An outer join emits every row of its preserved side whether or not it
    matched, so its output cannot fall below that side's cardinality.

    Unique keys on both sides make the floor bite: the inner estimate is
    `100 * 1000 / 1000 = 100`, which is already `|L|` and far below `|R|`.
    """
    var left = _source(
        [_column(String("k"), 0, 99, ndv=Approx.exact(100))], Approx.exact(100)
    )
    var right = _source(
        [_column(String("k"), 0, 999, ndv=Approx.exact(1000))],
        Approx.exact(1000),
    )

    var inner = left.join(right.copy(), [0], [0], JOIN_INNER)
    assert_equal(inner.estimate().rows.known().value(), 100)

    var l = left.join(right.copy(), [0], [0], JOIN_LEFT)
    assert_equal(l.estimate().rows.known().value(), 100)

    var r = left.join(right.copy(), [0], [0], JOIN_RIGHT)
    assert_equal(r.estimate().rows.known().value(), 1000)

    # max(100, 100) + max(100, 1000) - 100
    var full = left.join(right.copy(), [0], [0], JOIN_FULL)
    assert_equal(full.estimate().rows.known().value(), 1000)


def test_estimate_existence_filters_are_capped_by_the_side_they_project() raises:
    """`inner` is 2000: SEMI caps it at `|L|` = 100 and RIGHT_SEMI at `|R|` =
    1000, and each ANTI is the complement of its SEMI — floored at a row,
    since only a proof answers zero."""
    var sides = _join_sides()
    var semi = sides[0].join(sides[1].copy(), [0], [0], JOIN_SEMI)
    var anti = sides[0].join(sides[1].copy(), [0], [0], JOIN_ANTI)
    var right_semi = sides[0].join(sides[1].copy(), [0], [0], JOIN_RIGHT_SEMI)
    var right_anti = sides[0].join(sides[1].copy(), [0], [0], JOIN_RIGHT_ANTI)
    assert_equal(semi.estimate().rows.known().value(), 100)
    assert_equal(anti.estimate().rows.known().value(), 1)
    assert_equal(right_semi.estimate().rows.known().value(), 1000)
    assert_equal(right_anti.estimate().rows.known().value(), 1)


def test_estimate_join_against_nothing_is_exactly_nothing() raises:
    """A side with exactly zero rows makes the join exactly empty, however
    precisely the other side's rows are known."""
    var others: List[Approx] = [Approx.exact(1000), Approx.estimated(1000)]
    for ref other in others:
        var empty = _source([_column(String("k"), 0, 0)], Approx.exact(0))
        var right = _source(
            [_column(String("k"), 0, 49, ndv=Approx.exact(50))], other
        )
        var est = empty^.join(right^, [0], [0], JOIN_INNER).estimate()
        assert_true(est.rows.is_exact(), String("other side ", other))
        assert_equal(est.rows.known().value(), 0)


def test_estimate_is_the_same_under_a_mirrored_exchange() raises:
    """`k` over `(a, b)` and `k.mirror()` over `(b, a)` describe one answer.

    Not decoration: `JoinOrdering` may flip a join's build side, and a
    cardinality that moved when it did would let the search change its own
    input. Checked for all eight supported kinds rather than for the
    convenient ones — `JOIN_RIGHT_SEMI` exists precisely so the two that used
    to have no mirror now do.
    """
    var sides = _join_sides()
    var kinds: List[JoinKind] = [
        JOIN_INNER,
        JOIN_LEFT,
        JOIN_RIGHT,
        JOIN_FULL,
        JOIN_SEMI,
        JOIN_ANTI,
        JOIN_RIGHT_SEMI,
        JOIN_RIGHT_ANTI,
    ]
    for ref k in kinds:
        var here = sides[0].join(sides[1].copy(), [0], [0], k).estimate()
        var there = (
            sides[1].join(sides[0].copy(), [0], [0], k.mirror()).estimate()
        )
        assert_equal(
            here.rows,
            there.rows,
            String("kind ", k, ": the exchange moved the cardinality"),
        )
        assert_equal(len(here.columns), len(there.columns))


def test_estimate_does_not_move_with_the_build_side() raises:
    """A physical choice is invisible to an estimate, by construction:
    `build_side` never reaches `Estimate.joined`."""
    var sides = _join_sides()
    var here = sides[0].join(
        sides[1].copy(), [0], [0], JOIN_INNER, build_side=BUILD_LEFT
    )
    var there = sides[0].join(
        sides[1].copy(), [0], [0], JOIN_INNER, build_side=BUILD_RIGHT
    )
    assert_equal(here.estimate().rows, there.estimate().rows)


def test_estimate_join_output_columns_follow_the_schema() raises:
    """One column per output field: an inner key both sides call `k` is one,
    an outer join's two `k`s are two, and each of the four existence filters
    emits one side alone."""
    var sides = _join_sides()

    var inner = sides[0].join(sides[1].copy(), [0], [0], JOIN_INNER)
    assert_equal(len(inner.estimate().columns), 1)

    var left = sides[0].join(sides[1].copy(), [0], [0], JOIN_LEFT)
    assert_equal(len(left.estimate().columns), 2)
    assert_equal(left.estimate().columns[1].name, "k_right")

    var semi = sides[0].join(sides[1].copy(), [0], [0], JOIN_SEMI)
    assert_equal(len(semi.estimate().columns), 1)

    var right_semi = sides[0].join(sides[1].copy(), [0], [0], JOIN_RIGHT_SEMI)
    assert_equal(len(right_semi.estimate().columns), 1)


def test_estimate_a_join_with_an_unknown_side_has_no_estimate_or_cost() raises:
    """Unknown rows on one side leave the join's rows unknown and its cost
    unknown under either build side, so neither side can win."""
    var right = _source(
        [_column(String("k"), 0, 49, ndv=Approx.exact(50))], Approx.exact(1000)
    )
    var plan = _bare_scan().join(right.copy(), [0], [0], JOIN_INNER)
    var flipped = _bare_scan().join(right^, [0], [0], JOIN_INNER, BUILD_RIGHT)
    assert_false(plan.estimate().rows.is_known())
    assert_false(plan.cost().total().is_known())
    assert_false(flipped.cost().total().is_known())


def test_estimate_join_keeps_distinct_counts_for_the_join_above() raises:
    """A join's output still knows how many distinct values each column holds,
    so the join above it divides by a real domain.

    `A ⋈ B` on `ak = bk` is 100 * 1,000 / 10 = 10,000 rows; `bj` still holds
    50 values, so joining `C` on `bj = cj` is 10,000 * 500 / 50 = 100,000.
    Dropping the count, as `joined` once did, made `bj`'s domain the 10,000
    rows and the second join 500 — a 200-fold underestimate one join up.
    """
    var a = _source(
        [_column(String("ak"), 0, 999, ndv=Approx.exact(10))],
        Approx.exact(100),
    )
    var b = _source(
        [
            _column(String("bk"), 0, 999, ndv=Approx.exact(10)),
            _column(String("bj"), 0, 999, ndv=Approx.exact(50)),
        ],
        Approx.exact(1000),
    )
    var c = _source(
        [_column(String("cj"), 0, 999, ndv=Approx.exact(50))],
        Approx.exact(500),
    )
    var ab = a.join(b^, [0], [0], JOIN_INNER)
    assert_equal(ab.estimate().rows.known().value(), 10_000)
    assert_equal(ab.estimate().columns[2].ndv.known().value(), 50)
    var abc = ab.join(c^, [2], [0], JOIN_INNER)
    assert_equal(abc.estimate().rows.known().value(), 100_000)


def test_estimate_a_matched_key_takes_the_smaller_domain_and_no_null() raises:
    """Every row of an inner join matched, so each key value is on both
    sides: 10 distinct against 50 leaves at most 10 on the one `k` the join
    emits, and it cannot be NULL. A LEFT join keeps unmatched left rows, so
    its left key keeps its own counts, and pads the right side with NULLs, so
    the right key's null count is no longer known."""
    var sides = _join_sides()
    var inner = sides[0].join(sides[1].copy(), [0], [0], JOIN_INNER).estimate()
    assert_equal(len(inner.columns), 1)
    assert_equal(inner.columns[0].ndv.known().value(), 10)
    assert_true(inner.columns[0].nulls == Approx.exact(0))

    var left = sides[0].join(sides[1].copy(), [0], [0], JOIN_LEFT).estimate()
    assert_equal(left.columns[0].ndv.known().value(), 10)
    assert_equal(left.columns[1].ndv.known().value(), 50)
    assert_true(left.columns[0].nulls == Approx.exact(0))
    assert_false(left.columns[1].nulls.is_known())


def test_estimate_integer_bounds_cap_the_distinct_count() raises:
    """No distinct count, but bounds `[0, 9]`: an integer column cannot hold
    more than ten values, so the many-to-many join above is seen for what it
    is — 1,000 * 1,000 / 10 — without anyone having counted."""
    var plan = _source([_column(String("k"), 0, 9)], Approx.exact(1000)).join(
        _source([_column(String("k"), 0, 9)], Approx.exact(1000)),
        [0],
        [0],
        JOIN_INNER,
    )
    assert_equal(plan.estimate().rows.known().value(), 100_000)

    # and a recorded count larger than the range is capped by it
    var over = _column(String("k"), 0, 9, ndv=Approx.exact(50))
    assert_equal(over.max_distinct(Approx.exact(1000)).known().value(), 10)


def _bounded(lo: DynScalar, hi: DynScalar) -> ColumnEstimate:
    return ColumnEstimate(String("c"), lo.copy(), hi.copy())


def test_estimate_span_counts_integer_values_between_the_bounds() raises:
    """`max - min + 1` for every integer-stored dtype, a date included."""
    var i32 = _bounded(
        Int32Scalar(Scalar[int32.native](-5)).to_dyn(),
        Int32Scalar(Scalar[int32.native](10)).to_dyn(),
    )
    assert_equal(i32.span().known().value(), 16)

    var days = _bounded(
        Date32Scalar(Optional(Int32(19000)), date32()).to_dyn(),
        Date32Scalar(Optional(Int32(19364)), date32()).to_dyn(),
    )
    assert_equal(days.span().known().value(), 365)

    var ts = _bounded(
        TimestampScalar(Optional(Int64(-1)), timestamp(second)).to_dyn(),
        TimestampScalar(Optional(Int64(1)), timestamp(second)).to_dyn(),
    )
    assert_equal(ts.span().known().value(), 3)


def test_estimate_span_saturates_rather_than_wrapping() raises:
    """The full `uint64` and `int64` ranges hold 2^64 values, which neither
    fits an `Int` nor may wrap to a small one: it saturates."""
    var u = _bounded(
        UInt64Scalar(Scalar[uint64.native](0)).to_dyn(),
        UInt64Scalar(Scalar[uint64.native].MAX).to_dyn(),
    )
    assert_equal(u.span().known().value(), Int.MAX)

    var i = _bounded(
        Int64Scalar(Scalar[int64.native].MIN).to_dyn(),
        Int64Scalar(Scalar[int64.native].MAX).to_dyn(),
    )
    assert_equal(i.span().known().value(), Int.MAX)


def test_estimate_span_is_unknown_without_integer_bounds() raises:
    """A float's range holds no countable set of values, a string's none an
    integer can express, and inverted or missing bounds describe nothing."""
    var floats = _bounded(
        Float64Scalar(Scalar[float64.native](0.0)).to_dyn(),
        Float64Scalar(Scalar[float64.native](1.0)).to_dyn(),
    )
    assert_false(floats.span().is_known())

    var strings = _bounded(
        StringScalar(String("a")).to_dyn(), StringScalar(String("z")).to_dyn()
    )
    assert_false(strings.span().is_known())

    var inverted = _bounded(
        Int32Scalar(Scalar[int32.native](10)).to_dyn(),
        Int32Scalar(Scalar[int32.native](-5)).to_dyn(),
    )
    assert_false(inverted.span().is_known())

    assert_false(ColumnEstimate(String("c")).span().is_known())


def test_estimate_a_nullable_group_key_adds_the_null_group() raises:
    """GROUP BY puts every NULL key in one group of its own, which a distinct
    count does not include: three values plus NULL is four groups, and three
    once the key provably holds no NULL."""
    var nullable = ColumnEstimate(
        String("a"),
        Int64Scalar(Scalar[int64.native](0)).to_dyn(),
        Int64Scalar(Scalar[int64.native](9)).to_dyn(),
        ndv=Approx.exact(3),
        width=Approx.exact(8),
    )
    var input = Estimate(Approx.exact(100), [nullable.copy()])
    var s = schema([field("a", int64)])
    assert_equal(input.grouped(["a"], s).rows.known().value(), 4)

    var total = nullable.copy()
    total.nulls = Approx.exact(0)
    var no_nulls = Estimate(Approx.exact(100), [total^])
    assert_equal(no_nulls.grouped(["a"], s).rows.known().value(), 3)


def test_estimate_a_pruned_scan_estimates_only_its_columns() raises:
    """Narrowing a scan keeps its statistics whole, but its estimate must
    describe the columns it still reads: a dropped column's width is not paid
    by a join that never materialises it."""
    var src = _source(
        [_column(String("a"), 0, 9), _column(String("b"), 0, 9)],
        Approx.exact(100),
    )
    ref scan = src.get[ParquetScan]()
    var narrowed: DynRelation = scan.with_schema(schema([field("a", int64)]))
    var est = narrowed.estimate()
    assert_equal(len(est.columns), 1)
    assert_equal(est.columns[0].name, "a")
    assert_equal(est.row_width().known().value(), 8)


def test_estimate_a_rename_keeps_the_columns_summary() raises:
    """`a AS x` holds the same values as `a`, so the null count survives under
    the new name — the shape of every input the SQL frontend builds."""
    var est = table(_batch()).rename(["a"], ["x"]).estimate()
    assert_equal(est.columns[0].name, "x")
    assert_equal(est.columns[0].nulls.known().value(), 1)


def test_estimate_widths_follow_from_every_fixed_shape() raises:
    """A width is a property of the type wherever every value has one size:
    a dictionary holds only its index per row, a null column nothing, a struct
    and a fixed-size list what their parts hold. A list does not."""
    assert_equal(
        ColumnEstimate.width_of(FixedSizeBinaryType(4)).known().value(), 4
    )
    assert_equal(
        ColumnEstimate.width_of(dictionary(int32, string)).known().value(), 4
    )
    assert_equal(ColumnEstimate.width_of(null).known().value(), 0)
    assert_equal(
        ColumnEstimate.width_of(struct_(field("a", int64), field("b", int32)))
        .known()
        .value(),
        12,
    )
    assert_equal(
        ColumnEstimate.width_of(fixed_size_list_(int16, 3)).known().value(), 6
    )
    assert_equal(ColumnEstimate.width_of(list_(int64)).known().value(), 8)
    assert_false(ColumnEstimate.width_of(list_(int64)).is_exact())


def test_estimate_counts_saturate_instead_of_wrapping() raises:
    """A search builds intermediates nobody wrote, and the largest must price
    as the largest: a wrapped product would clamp to zero and win."""
    var huge = Approx.estimated(Int.MAX // 2 + 1)
    assert_equal((huge * Approx.exact(4)).known().value(), Int.MAX)
    assert_equal((huge + huge).known().value(), Int.MAX)
    assert_equal(huge.times(3).known().value(), Int.MAX)
    # A percentage of a near-saturated count is still that percentage.
    assert_equal(
        huge.scaled(DEFAULT_SELECTIVITY).known().value() // DEFAULT_SELECTIVITY,
        (Int.MAX // 2 + 1) // 100,
    )


def test_estimate_a_filter_over_nothing_is_still_exactly_nothing() raises:
    """An input proven empty stays proven empty, whatever the predicate says
    about bounds that describe no row."""
    var empty = _source([_column(String("a"), 0, 9)], Approx.exact(0))
    var est = empty.filter(col("a", int64) > lit(5, int64)).estimate()
    assert_true(est.rows == Approx.exact(0))


# ---------------------------------------------------------------------------
# analyze -- statistics a caller asked for
# ---------------------------------------------------------------------------
def test_estimate_analyze_summarizes_an_in_memory_table() raises:
    """Bounds skip the NULL, the distinct count is the values present, and the
    counts stay exact."""
    var est = table(_batch()).analyze().estimate()
    assert_true(est.rows == Approx.exact(4))
    assert_equal(est.columns[0].min.as_int64().value(), 1)
    assert_equal(est.columns[0].max.as_int64().value(), 4)
    assert_equal(est.columns[0].nulls.known().value(), 1)
    assert_equal(est.columns[0].ndv.known().value(), 3)
    assert_equal(est.columns[1].ndv.known().value(), 4)


def test_estimate_analyze_counts_distinct_values_closely() raises:
    """A HyperLogLog sketch, not an exact count — but within 2% at 100,000
    rows over 50,000 values."""
    var values = List[Optional[Int]](capacity=100_000)
    for i in range(100_000):
        values.append(i % 50_000)
    var est = summarize(
        record_batch([array(values^, int64).to_dyn()], names=["k"])
    )
    var ndv = est.columns[0].ndv.known().value()
    assert_true(49_000 <= ndv and ndv <= 51_000, String("ndv ", ndv))


def test_estimate_analyze_skips_nan_in_the_bounds() raises:
    """A NaN maximum would prove `x > 2` false and prune a row that matches
    it, so a bound is the largest *number*."""
    var values: List[Optional[Float64]] = [
        Float64(1.0),
        nan[DType.float64](),
        Float64(3.0),
    ]
    var est = summarize(
        record_batch([array(values^, float64).to_dyn()], names=["x"])
    )
    assert_equal(est.columns[0].max.as_float64().value(), 3.0)
    assert_equal(est.columns[0].min.as_float64().value(), 1.0)


def test_estimate_analyze_measures_a_slices_strings() raises:
    """Two strings of three bytes each, sliced out of a batch whose other
    rows are long: the width is theirs, three bytes and a four-byte offset,
    not the shared buffer's average."""
    var batch = record_batch(
        [
            array(
                ["xxxxxxxxxxxxxxxxxxxx", "abc", "def", "yyyyyyyyyyyyyyyy"]
            ).copy()
        ],
        names=["s"],
    )
    var est = summarize(batch.slice(1, 2))
    assert_equal(est.columns[0].width.known().value(), 7)
    assert_true(est.row_width().is_known())


def test_estimate_analyze_statistics_survive_column_pruning() raises:
    """`ColumnPruning` narrows an in-memory source to what the plan reads,
    and must keep what `analyze` found in it — dropping it costs an
    optimization rather than an answer, so nothing else would fail."""
    var plan = table(_batch()).analyze().select(["a"])
    var optimized = plan.optimize[AllRules]()
    var est = optimized.estimate()
    assert_equal(len(est.columns), 1)
    assert_equal(est.columns[0].ndv.known().value(), 3)


def test_estimate_analyze_makes_a_many_to_many_join_visible() raises:
    """1,000 rows over 25 keys joined to 1,000 more: blind, the join is
    estimated at 1,000 — the row count stands in for the domain — and
    analysed it is 40,000, which is what it produces."""
    var left = List[Optional[Int]](capacity=1_000)
    var right = List[Optional[Int]](capacity=1_000)
    for i in range(1_000):
        left.append(i % 25)
        right.append((i * 7) % 25)
    var l = table(record_batch([array(left^, int64).to_dyn()], names=["lk"]))
    var r = table(record_batch([array(right^, int64).to_dyn()], names=["rk"]))

    var blind = l.join(r.copy(), [0], [0], JOIN_INNER)
    assert_equal(blind.estimate().rows.known().value(), 1_000)
    var informed = l.analyze().join(r.analyze(), [0], [0], JOIN_INNER)
    assert_equal(informed.estimate().rows.known().value(), 40_000)


def _strings_fixture(path: String, dictionary: Bool) raises:
    """1,000 five-byte strings over 25 values, in one row group."""
    var values = List[Optional[String]](capacity=1_000)
    for i in range(1_000):
        values.append(String("v") + String(1_000 + i % 25))
    var batch = record_batch([array(values^).copy()], names=["s"])
    var w = FileWriter(
        FileSink(path), Compression.UNCOMPRESSED, use_dictionary=dictionary
    )
    w.write(Table.from_batches(Schema(copy=batch.schema), [batch.copy()]))


def _footer_estimate(path: String, s: Schema) raises -> Estimate:
    """What `analyze` reads off the footer of the file at `path`."""
    var plan = scan(path.copy(), s.copy()).analyze()
    return plan.get[ParquetScan]().statistics.value()[].copy()


def test_estimate_a_footer_gives_a_string_column_a_width() raises:
    """Five bytes and a four-byte length prefix per value, read off the
    chunk's uncompressed size for PLAIN and off the dictionary entries for a
    dictionary-encoded chunk. Both are estimates — page headers and levels
    ride along — so each is asserted within a few bytes of nine."""
    with ScratchDir() as dir:
        var s = schema([field("s", string)])
        for dictionary in [False, True]:
            var path = join(dir, String("strings_", dictionary, ".parquet"))
            _strings_fixture(path, dictionary)
            var est = _footer_estimate(path, s)
            var width = est.columns[0].width
            assert_false(width.is_exact())
            assert_true(
                9 <= width.known().value() and width.known().value() <= 13,
                String("dictionary=", dictionary, ": width ", width),
            )


def test_estimate_analyze_reads_a_scans_footer() raises:
    """A scan analysed reads its footer, and nothing else: the statistics
    are exact row and null counts and a width for its string column."""
    with ScratchDir() as dir:
        var path = join(dir, "analyzed.parquet")
        _strings_fixture(path, False)
        var plan = scan(path.copy(), schema([field("s", string)])).analyze()
        var est = plan.estimate()
        assert_true(est.rows == Approx.exact(1_000))
        assert_true(est.row_width().is_known())


def test_estimate_analyze_reads_every_file_of_a_scan() raises:
    """A scan of two files is analysed from both footers: their rows add up,
    and the string column's width is measured over both."""
    with ScratchDir() as dir:
        var a = join(dir, "a.parquet")
        var b = join(dir, "b.parquet")
        _strings_fixture(a, False)
        _strings_fixture(b, True)
        var s = schema([field("s", string)])
        var est = _footer_estimate_of([a.copy(), b.copy()], s)
        assert_true(est.rows == Approx.exact(2_000))
        assert_true(est.columns[0].nulls == Approx.exact(0))
        var width = est.columns[0].width
        assert_false(width.is_exact())
        assert_true(
            9 <= width.known().value() and width.known().value() <= 13,
            String("width ", width),
        )


def _footer_estimate_of(var paths: List[String], s: Schema) raises -> Estimate:
    """What `analyze` reads off the footers of the files at `paths`."""
    var plan = scan(paths^, s.copy()).analyze()
    return plan.get[ParquetScan]().statistics.value()[].copy()


# ---------------------------------------------------------------------------
# Source statistics from a real footer
# ---------------------------------------------------------------------------
def test_estimate_reads_a_parquet_footer_without_reading_the_file() raises:
    """`Index.from_parquet` decoded the footer; `Estimate.from_index` only
    reduces what it holds.

    Row counts and null counts come out **exact** — a footer states both — and
    the bounds reduce across row groups in the column's own dtype.

    Distinct counts stay unknown here, and the reason is the *writer*, not the
    reader: pyarrow 23 leaves `Statistics.distinct_count` absent even for a
    dictionary-encoded column (`has_distinct_count` is False), so there is
    nothing in this footer to reduce. An absent statistic must not become a
    number — answering `rows` instead would be an upper bound wearing a
    figure's clothes, and `max_distinct` already makes that assumption where
    its caller can see it. The companion case below writes the same shape with
    marrow's own writer and does get one.
    """
    with ScratchDir() as dir:
        var path = join(dir, "marrow_estimate_footer.parquet")
        var pa = Python.import_module("pyarrow")
        var pq = Python.import_module("pyarrow.parquet")
        var a = Python.list()
        var b = Python.list()
        for i in range(400):
            a.append(i)
            b.append(i % 7)
        var tbl = pa.table(Python.dict(a=pa.array(a), b=pa.array(b)))
        pq.write_table(tbl, path, row_group_size=100, compression="none")

        var f = ParquetFile(path)
        var s = schema([field("a", int64), field("b", int64)])
        var est = Estimate.from_index(Index.from_parquet(f), s)

        assert_true(est.rows.is_exact())
        assert_equal(est.rows.known().value(), 400)
        assert_equal(est.columns[0].min.as_int64().value(), 0)
        assert_equal(est.columns[0].max.as_int64().value(), 399)
        assert_equal(est.columns[0].nulls.known().value(), 0)
        assert_false(est.columns[0].ndv.is_known())


def test_estimate_scan_statistics_survive_the_optimizer() raises:
    """A rewrite that rebuilds a scan must carry its statistics, for the same
    reason it must carry its pruners: dropping them costs an optimization
    rather than an answer, so nothing else would fail."""
    var src = _source([_column(String("a"), 0, 9)], Approx.exact(100))
    var plan = src.filter(col("a", int64) > lit(5, int64))
    var optimized = plan.optimize[AllRules]()

    assert_true(optimized.estimate().rows.is_known())
    assert_equal(optimized.estimate().rows.known().value(), 40)


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------
def test_estimate_cost_is_a_readable_sum() raises:
    """Four rows of two `int64` columns: 4 rows moved, 64 bytes built, no
    comparisons — and the filter above adds one evaluation per row."""
    var source = table(_batch())
    assert_equal(source.cost().rows.known().value(), 4)
    assert_equal(source.cost().bytes.known().value(), 64)
    assert_equal(source.cost().compares.known().value(), 0)
    assert_equal(source.cost().total().known().value(), 68)

    var filtered = source.filter(col("a", int64) > lit(2, int64))
    assert_equal(filtered.cost().rows.known().value(), 8)
    assert_equal(filtered.cost().total().known().value(), 72)


def test_estimate_cost_prefers_filtering_before_sorting() raises:
    """The comparison the model exists to make.

    Sorting 100 rows costs `n log n` comparisons and materialises all of them;
    sorting the ~20 that survive a filter costs less on both counts. The two
    plans compute the same answer, so a model that could not tell them apart
    would be worth nothing.
    """
    var src = _source([_column(String("a"), 0, 9)], Approx.exact(100))

    var sort_then_filter = (
        src.sort_by([col("a", int64)], [True])
        .filter(col("a", int64) > lit(5, int64))
        .cost()
    )
    var filter_then_sort = (
        src.filter(col("a", int64) > lit(5, int64))
        .sort_by([col("a", int64)], [True])
        .cost()
    )

    assert_true(sort_then_filter.total().is_known())
    assert_true(filter_then_sort.total().is_known())
    assert_true(
        filter_then_sort.total().known().value()
        < sort_then_filter.total().known().value(),
        String(filter_then_sort) + " vs " + String(sort_then_filter),
    )


def test_estimate_cost_of_an_unestimable_plan_is_unknown() raises:
    """Not zero — a node nobody can estimate must not score better than every
    node that can, which is what a free default would make it do."""
    var plan = _bare_scan().sort_by([col("a", int64)], [True])
    assert_false(plan.cost().total().is_known())


def test_estimate_cost_says_which_side_should_be_the_build_side() raises:
    """`Cost.hash_join` is asymmetric, and that asymmetry is the whole point.

    100 rows against 1000: hashing the small side and streaming the large one
    must score better than the reverse, and the difference is exactly the
    materialisation term — `hash_build` charges the bytes it holds where
    `hash_probe` charges nothing but a comparison. A chain's `cost()` reads
    each join's own side, so a tree that made the expensive choice scores as
    having made it.
    """
    var sides = _join_sides()
    var plan = sides[0].join(sides[1].copy(), [0], [0], JOIN_INNER)
    var flipped = sides[0].join(
        sides[1].copy(), [0], [0], JOIN_INNER, BUILD_RIGHT
    )

    var build_small = plan.cost().total().known().value()
    var build_large = flipped.cost().total().known().value()
    assert_true(
        build_small < build_large,
        String(
            "indexing 100 rows should beat indexing 1000: ",
            build_small,
            " vs ",
            build_large,
        ),
    )


def test_estimate_reads_a_distinct_count_from_marrows_own_footer() raises:
    """`Estimate.from_index` reduces the distinct count marrow's own writer
    records for a dictionary-encoded chunk into the column's `ndv`.

    The fixture is written by marrow rather than pyarrow **because pyarrow
    writes no distinct count at all** — the case above pins that. Four row
    groups of 100 rows over a key with 25 distinct values: every group holds
    all 25, so every chunk's statistic is 25 and the reduction answers 25.

    `b` is the contrast in the same file: 400 distinct values over four
    groups, 100 to a group and written in order, so the groups' ranges are
    disjoint, no value is in two of them, and the counts add up to the truth.

    Never **exact**: the per-chunk figure is a dictionary size, which
    parquet-format lets a writer overstate, and the maximum is a lower bound
    on the column regardless.
    """
    with ScratchDir() as dir:
        var path = join(dir, "marrow_estimate_ndv.parquet")
        var keys = List[Optional[Int]](capacity=400)
        var uniq = List[Optional[Int]](capacity=400)
        for i in range(400):
            keys.append(i % 25)
            uniq.append(i)
        var b = record_batch(
            [array(keys, int64).copy(), array(uniq, int64).copy()],
            names=["k", "b"],
        )
        var s = Schema(copy=b.schema)
        var w = FileWriter(FileSink(path), Compression.UNCOMPRESSED)
        w.write(Table.from_batches(s.copy(), [b^]), row_group_size=100)

        var f = ParquetFile(path)
        var index = Index.from_parquet(f)
        assert_equal(index.chunks(), 4)
        assert_equal(
            index.columns[index.find("k")].distinct_counts, [25, 25, 25, 25]
        )
        assert_equal(
            index.columns[index.find("b")].distinct_counts,
            [100, 100, 100, 100],
        )

        var est = Estimate.from_index(index, s)
        assert_true(est.columns[0].ndv.is_known())
        assert_false(est.columns[0].ndv.is_exact())
        assert_equal(est.columns[0].ndv.known().value(), 25)
        # `b` is written in order, a hundred values per row group, so the
        # groups are disjoint and their counts add up.
        assert_equal(est.columns[1].ndv.known().value(), 400)


comptime _DICT_FALLBACK_DISTINCT = 135_000
"""Distinct `int64` values in one row group — enough to force PLAIN.

`writer._DICT_PAGE_LIMIT` is one mebibyte and a dictionary page holds each
distinct value PLAIN-encoded, so an `int64` column falls back above 131,072
distinct values. A margin rather than the exact threshold: what the fixture
needs is that one chunk dictionary-encodes and another does not, not where the
boundary sits.
"""


def test_estimate_reduces_across_mixed_dictionary_and_plain_chunks() raises:
    """Row groups that disagree on encoding: marrow records a distinct count
    only for a dictionary-encoded chunk, so a high-cardinality group falls
    back to PLAIN and records none. The column still answers the recorded
    count, read from a real file rather than from hand-written lists."""
    with ScratchDir() as dir:
        var path = join(dir, "marrow_estimate_mixed_encoding.parquet")
        var n = _DICT_FALLBACK_DISTINCT
        var vals = List[Optional[Int]](capacity=n + 100)
        for i in range(n):
            vals.append(i)
        for i in range(100):
            vals.append(i % 7)
        var b = record_batch([array(vals^, int64).to_dyn()], names=["v"])
        var s = Schema(copy=b.schema)
        var w = FileWriter(FileSink(path), Compression.UNCOMPRESSED)
        w.write(Table.from_batches(s.copy(), [b^]), row_group_size=n)

        var index = Index.from_parquet(ParquetFile(path))
        assert_equal(index.chunks(), 2)
        # The mixed shape itself: the big group fell back to PLAIN and recorded
        # nothing, the hundred-row group dictionary-encoded and recorded seven.
        assert_equal(index.columns[index.find("v")].distinct_counts, [-1, 7])

        var est = Estimate.from_index(index, s)
        assert_true(est.rows.is_exact())
        assert_equal(est.rows.known().value(), n + 100)
        assert_true(est.columns[0].ndv.is_known())
        assert_false(est.columns[0].ndv.is_exact())
        assert_equal(est.columns[0].ndv.known().value(), 7)


def test_estimate_reduces_per_chunk_distinct_counts_with_the_maximum() raises:
    """An estimate takes the maximum distinct count over the chunks that recorded
    one: `[3, 7, 5]` answers 7 rather than the sum, `[-1, 7]` answers 7, and
    no recorded count answers unknown rather than zero."""
    assert_equal(_ndv_of([3, 7, 5]).known().value(), 7)
    assert_equal(_ndv_of([-1, 7]).known().value(), 7)
    assert_equal(_ndv_of([7, -1]).known().value(), 7)
    assert_false(_ndv_of([-1, -1]).is_known())
    assert_false(_ndv_of(List[Int]()).is_known())
    # and it is a claim, not a fact, whatever the chunks said
    assert_false(_ndv_of([3]).is_exact())


def _page_bounds(rows: Int, lo: Int, hi: Int) raises -> PageBounds:
    """One `int64` data page with both bounds recorded."""
    return PageBounds(
        rows,
        False,
        Optional(Int64Scalar(Scalar[int64.native](lo)).to_dyn()),
        Optional(Int64Scalar(Scalar[int64.native](hi)).to_dyn()),
    )


def test_estimate_from_a_page_index_has_no_distinct_count() raises:
    """A page index records no distinct counts, so an estimate from one has
    `ndv` unknown; padding them with zeros would instead give every join a
    zero denominator and so no estimate. Bounds and row counts are asserted
    too, since "unknown" is also what an empty index would answer."""
    var index = Index.from_pages(
        String("a"),
        [_page_bounds(10, 0, 9), _page_bounds(10, 10, 19)],
        20,
    )
    assert_equal(index.chunks(), 2)
    assert_equal(index.columns[0].distinct_counts, List[Int]())
    # and the column really is described — the bounds came through
    assert_equal(index.columns[0].null_counts, [-1, -1])

    var est = Estimate.from_index(index, schema([field("a", int64)]))
    assert_true(est.rows.is_exact())
    assert_equal(est.rows.known().value(), 20)
    assert_equal(est.columns[0].min.as_int64().value(), 0)
    assert_equal(est.columns[0].max.as_int64().value(), 19)
    # A page index records no null count either, so neither count is provable.
    assert_false(est.columns[0].nulls.is_known())
    assert_false(est.columns[0].ndv.is_known())

    # And the fallback is spelled where its caller can see it: `max_distinct`
    # answers the row count, as an estimate, rather than a fabricated `ndv`.
    var fallback = est.columns[0].max_distinct(est.rows)
    assert_equal(fallback.known().value(), 20)
    assert_false(fallback.is_exact())


def test_estimate_join_cardinality_changes_when_ndv_is_known() raises:
    """The reason the hop was wired: a recorded `ndv` changes the answer.

    Two 1,000-row sides joined on a key with 10 distinct values is a
    many-to-many blow-up — 100,000 rows. Without a distinct count, and with
    bounds too wide for `span` to bite, `max_distinct` falls back to the row
    count, the denominator becomes 1,000 and the join is predicted to *shrink*
    to 1,000: the blow-up reads as the cheapest thing in the plan, which is
    exactly backwards.
    """
    var blind = _source(
        [_column(String("k"), 0, 999_999)], Approx.exact(1000)
    ).join(
        _source([_column(String("k"), 0, 999_999)], Approx.exact(1000)),
        [0],
        [0],
        JOIN_INNER,
    )
    assert_equal(blind.estimate().rows.known().value(), 1000)

    var informed = _source(
        [_column(String("k"), 0, 9, ndv=Approx.estimated(10))],
        Approx.exact(1000),
    ).join(
        _source(
            [_column(String("k"), 0, 9, ndv=Approx.estimated(10))],
            Approx.exact(1000),
        ),
        [0],
        [0],
        JOIN_INNER,
    )
    assert_equal(informed.estimate().rows.known().value(), 100_000)


# ---------------------------------------------------------------------------
# End to end: a footer's distinct count changes the plan
# ---------------------------------------------------------------------------
comptime _A_FILE = "marrow_reassoc_a.parquet"
comptime _B_FILE = "marrow_reassoc_b.parquet"
comptime _C_FILE = "marrow_reassoc_c.parquet"


def _reassoc_write(batch: RecordBatch, path: String, row_group: Int) raises:
    """One table through marrow's own writer — the only writer that emits a
    `Statistics.distinct_count` at all."""
    var s = Schema(copy=batch.schema)
    var w = FileWriter(FileSink(path), Compression.UNCOMPRESSED)
    w.write(Table.from_batches(s^, [batch.copy()]), row_group_size=row_group)


def _reassoc_fixture(dir: String) raises:
    """`A ⋈ B` many-to-many on `ak`, `B ⋈ C` nearly empty on `bc`.

    Deliberately the smallest shape that separates the two arms: 25,000 rows
    of `A` over 25 `ak` values against 1,000 rows of `B` over the same 25, so
    the intermediate really is 1,000,000 rows, where `C` matches 25 rows of
    `B` and the answer is 25,000.
    """
    var ak = List[Optional[Int]](capacity=25_000)
    for i in range(25_000):
        ak.append(i % 25)
    # Five row groups, so the per-chunk distinct counts are reduced rather
    # than read: every group holds all 25 values, and the maximum is right.
    _reassoc_write(
        record_batch([array(ak^, int64).to_dyn()], names=["ak"]),
        join(dir, _A_FILE),
        5_000,
    )

    var bak = List[Optional[Int]](capacity=1_000)
    var bbc = List[Optional[Int]](capacity=1_000)
    for i in range(1_000):
        bak.append(i % 25)
        bbc.append(i)
    _reassoc_write(
        record_batch(
            [array(bak^, int64).to_dyn(), array(bbc^, int64).to_dyn()],
            names=["ak", "bc"],
        ),
        join(dir, _B_FILE),
        1_000,
    )

    var cbc = List[Optional[Int]](capacity=1_000)
    var cv = List[Optional[Int]](capacity=1_000)
    for i in range(1_000):
        cbc.append(i if i < 25 else 1_000 + i)
        cv.append(i)
    _reassoc_write(
        record_batch(
            [array(cbc^, int64).to_dyn(), array(cv^, int64).to_dyn()],
            names=["bc", "cv"],
        ),
        join(dir, _C_FILE),
        1_000,
    )


def _without_ndv(est: Estimate) raises -> Estimate:
    """`est` with nothing known about how many values its columns hold:
    `ndv` unknown and the bounds dropped too, since `span` reads a distinct
    count off integer bounds. Rows, null counts and widths stay intact.

    The blind arm spelled as a subtraction from the informed one, so the two
    cannot differ in anything but what counts values.
    """
    var cols = List[ColumnEstimate](capacity=len(est.columns))
    for ref c in est.columns:
        cols.append(
            ColumnEstimate(
                c.name.copy(),
                nulls=c.nulls,
                ndv=Approx.unknown(),
                width=c.width,
            )
        )
    return Estimate(est.rows, cols^)


def _reassoc_source(path: String, s: Schema, ndv: Bool) raises -> DynRelation:
    var est = Estimate.from_index(Index.from_parquet(ParquetFile(path)), s)
    if not ndv:
        est = _without_ndv(est)
    var node = ParquetScan(ScanPath(path.copy()), s.copy()).with_statistics(
        est^
    )
    var out: DynRelation = node^
    return out^


def _reassoc_pair(dir: String, ndv: Bool) raises -> DynRelation:
    """`A ⋈ B` on the key both call `ak`, over the fixture's own footers."""
    var a = schema([field("ak", int64)])
    var b = schema([field("ak", int64), field("bc", int64)])
    return _reassoc_source(join(dir, _A_FILE), a, ndv).join(
        _reassoc_source(join(dir, _B_FILE), b, ndv), [0], [0], JOIN_INNER
    )


def _reassoc_plan(dir: String, ndv: Bool) raises -> DynRelation:
    """`(A ⋈ B) ⋈ C`, left-deep as written: `ak, bc` joined to `C` on the
    `bc` both call by that name."""
    var c = schema([field("bc", int64), field("cv", int64)])
    return _reassoc_pair(dir, ndv).join(
        _reassoc_source(join(dir, _C_FILE), c, ndv), [1], [0], JOIN_INNER
    )


def test_estimate_a_footers_distinct_count_changes_the_chosen_plan() raises:
    """The whole hop, end to end, asserted as the thing it was wired for.

    `ColumnMetaData.distinct_count` -> `ColumnStatistics` -> `ColumnZones` ->
    `ColumnEstimate.ndv` -> `joined` -> the join search. Blind, `A ⋈ B` is
    estimated at 1,000 rows and the left-deep plan is kept; informed, it is
    estimated at 1,000,000 — which is what it actually produces — and `B ⋈ C`,
    25 rows, is joined first.
    """
    with ScratchDir() as dir:
        _reassoc_fixture(dir)

        var blind = _reassoc_plan(dir, ndv=False)
        var informed = _reassoc_plan(dir, ndv=True)

        # The cardinality the two arms disagree about.
        var blind_pair = _reassoc_pair(dir, ndv=False).estimate()
        var informed_pair = _reassoc_pair(dir, ndv=True).estimate()
        assert_equal(blind_pair.rows.known().value(), 1_000)
        assert_equal(informed_pair.rows.known().value(), 1_000_000)

        var blind_out = blind.optimize[AllRules]()
        var informed_out = informed.optimize[AllRules]()

        # The diff itself, on stderr rather than described in prose: `pytest -s`
        # shows it, and a failure below shows it anyway.
        print("\n[blind]    ", blind_out, file=stderr)
        print("[informed] ", informed_out, file=stderr)
        assert_true(
            informed_out.cost().total().known().value()
            < informed.optimize[NoRules]().cost().total().known().value(),
            "informed should reorder: " + String(informed_out),
        )
        # Same answer either way — the property the rewrite rests on.
        assert_true(blind_out.schema() == informed_out.schema())
        assert_equal(blind_out.execute().num_rows(), 25_000)
        assert_equal(informed_out.execute().num_rows(), 25_000)


def test_estimate_analyze_leaves_an_unbound_scan_path_alone() raises:
    """A scan whose path is a parameter this execution has not bound has no
    file to read yet: it is left without statistics rather than failing."""
    var s = schema([field("a", int64)])
    var plan = scan(param("src", string), s).analyze()
    assert_true(not plan.get[ParquetScan]().statistics)


def test_estimate_analyze_keeps_statistics_a_source_already_has() raises:
    """Analysing twice reads nothing twice: statistics a source carries,
    recorded or written by hand, are kept."""
    var node = InMemoryTable(_batch()).with_statistics(
        Estimate(Approx.exact(7), [])
    )
    var plan: DynRelation = node^
    assert_equal(plan.analyze().estimate().rows, Approx.exact(7))


def test_estimate_analyze_reaches_every_participant_of_a_chain() raises:
    """A chain's participants are its children, so `analyze` summarises each
    one."""
    var plan = table(_batch()).join(table(_batch()), [0], [0]).analyze()
    ref chain = plan.get[JoinChain]()
    for ref input in chain.inputs:
        assert_true(Bool(input[].get[InMemoryTable]().statistics))


def test_estimate_scaling_never_inverts_two_counts() raises:
    """A filter keeps a share of its input, so a larger input never filters
    to fewer rows — near saturation included."""
    var huge = Approx.estimated(Int.MAX).scaled(DEFAULT_SELECTIVITY)
    var large = Approx.estimated(1_000_000_000_000_000_000).scaled(
        DEFAULT_SELECTIVITY
    )
    assert_true(huge.known().value() >= large.known().value())
    assert_equal(Approx.estimated(1000).scaled(20).known().value(), 200)
    assert_equal(Approx.estimated(3).scaled(20).known().value(), 1)


def test_estimate_a_join_any_keeps_at_most_its_probe_rows() raises:
    """An inner `JOIN_ANY` keeps one match per probe row: hashing the left
    side, the right side's 100 rows bound the answer however many keys
    repeat."""
    var left = _source(
        [_column(String("k"), 0, 9, ndv=Approx.exact(10))], Approx.exact(1000)
    )
    var right = _source(
        [_column(String("k"), 0, 9, ndv=Approx.exact(10))], Approx.exact(100)
    )
    var all = left.join(right.copy(), [0], [0])
    assert_equal(all.estimate().rows.known().value(), 10_000)
    var any = left.join(
        right.copy(), [0], [0], JOIN_INNER, BUILD_LEFT, JOIN_ANY
    )
    assert_true(any.estimate().rows.known().value() <= 100)


def test_estimate_a_dictionary_fallback_chunk_has_a_plain_width() raises:
    """A writer that runs out of dictionary space falls back to PLAIN pages
    part-way through a chunk. Its `encoding_stats` says so, and the chunk's
    width is then its uncompressed size per value — never the dictionary
    formula, which would count the PLAIN pages as entries and multiply them by
    every value."""
    with ScratchDir() as dir:
        var path = join(dir, "marrow_estimate_fallback.parquet")
        var pa = Python.import_module("pyarrow")
        var pq = Python.import_module("pyarrow.parquet")
        var s = Python.list()
        for i in range(2000):
            s.append(String("value-", i, "-padding"))
        var tbl = pa.table(Python.dict(s=pa.array(s)))
        pq.write_table(
            tbl, path, compression="none", dictionary_pagesize_limit=256
        )
        var est = _footer_estimate(path, schema([field("s", string)]))
        var width = est.columns[0].width.known()
        assert_true(Bool(width), String(est.columns[0].width))
        # About 20 bytes a value and a 4-byte offset.
        assert_true(width.value() < 64, String(width.value()))


# ---------------------------------------------------------------------------
# What a filter keeps, and what a limit or a join leaves of a column
# ---------------------------------------------------------------------------
def test_estimate_an_equality_keeps_one_distinct_values_share() raises:
    """`a = 3` over ten values of `a` keeps a tenth of the rows, read off a
    histogram of `a` by the predicate's own `mask`; a range keeps the share of
    the range it covers. Neither is the flat default."""
    var src = _source(
        [_column(String("a"), 0, 9, Approx.estimated(10))], Approx.exact(1_000)
    )
    var eq = src.filter(col("a", int64) == lit(3, int64)).estimate()
    assert_equal(eq.rows.known().value(), 100)
    assert_false(eq.rows.is_exact())
    var low = src.filter(col("a", int64) < lit(3, int64)).estimate()
    assert_equal(low.rows.known().value(), 300)


def test_estimate_a_filter_keeps_counts_as_estimates() raises:
    """A filter that decides nothing keeps each column's null and distinct
    counts as estimates no larger than the rows left — and a column proven
    free of NULLs stays proven."""
    var src = _source(
        [_column(String("a"), 0, 9, Approx.estimated(10))], Approx.exact(1_000)
    )
    var est = src.filter(col("a", int64) != lit(3, int64)).estimate()
    assert_true(est.columns[0].nulls == Approx.exact(0))
    assert_equal(est.columns[0].ndv.known().value(), 10)


def test_estimate_a_limit_that_cannot_cut_changes_nothing() raises:
    """`LIMIT 100` over exactly four rows keeps them all, so every count stays
    exactly what it was; one that cuts leaves estimates."""
    var exact = table(_batch()).estimate()
    var whole = table(_batch()).limit(100).estimate()
    assert_true(whole.rows == Approx.exact(4))
    assert_true(whole.columns[0].nulls == exact.columns[0].nulls)
    assert_true(whole.columns[0].nulls.is_exact())
    var cut = table(_batch()).limit(2).estimate()
    assert_true(cut.rows == Approx.exact(2))
    assert_false(cut.columns[0].nulls.is_exact())


def test_estimate_an_anti_join_is_never_estimated_empty() raises:
    """Every customer may have an order, but only a proof says none is left:
    the complement is floored at one row."""
    var customers = Estimate(
        Approx.exact(1_000), [_column(String("c"), 0, 999)]
    )
    var orders = Estimate(Approx.exact(10_000), [_column(String("o"), 0, 999)])
    var anti = Estimate.joined(customers, orders, ["c"], ["o"], JOIN_ANTI)
    assert_true(anti.rows == Approx.estimated(1), String(anti))


def test_estimate_a_matched_key_takes_the_tighter_bounds() raises:
    """An inner join keeps only values both sides hold, so the key's bounds
    are the intersection of the two sides'."""
    var left = Estimate(Approx.exact(100), [_column(String("l"), 0, 99)])
    var right = Estimate(Approx.exact(10), [_column(String("r"), 50, 200)])
    var joined = Estimate.joined(left, right, ["l"], ["r"], JOIN_INNER)
    ref key = joined.columns[0]
    assert_equal(Int(key.min.as_int64().value()), 50)
    assert_equal(Int(key.max.as_int64().value()), 99)
    ref other = joined.columns[1]
    assert_equal(Int(other.min.as_int64().value()), 50)
    assert_equal(Int(other.max.as_int64().value()), 99)


def test_estimate_a_variable_width_column_takes_a_default_width() raises:
    """An unmeasured string or binary column is priced at a default width, as an estimate, so a plan reading one can still be costed.
    """
    var s = ColumnEstimate.width_of(string)
    assert_equal(s.known().value(), DEFAULT_STRING_WIDTH)
    assert_false(s.is_exact())
    assert_equal(
        ColumnEstimate.width_of(binary).known().value(), DEFAULT_BINARY_WIDTH
    )
    assert_equal(ColumnEstimate.width_of(list_(int64)).known().value(), 8)


def test_estimate_a_join_holding_its_probe_side_pays_for_it() raises:
    """A LEFT join built on its left side emits that side's unmatched rows
    only once every probe row is in, so it holds the probe side as well; built
    on its right side, it streams."""
    var left = Size(Approx.exact(100), Approx.exact(8))
    var right = Size(Approx.exact(1_000), Approx.exact(8))
    var held = Cost.hash_join(left, right, BUILD_LEFT, JOIN_LEFT)
    var streamed = Cost.hash_join(left, right, BUILD_LEFT, JOIN_INNER)
    assert_equal(held.bytes.known().value(), 800 + 8_000)
    assert_equal(streamed.bytes.known().value(), 800)


def test_estimate_disjoint_chunks_sum_their_distinct_counts() raises:
    """A key written in order, a chunk per range: no value is in two chunks,
    so the source's distinct count is the sum. Overlapping ranges keep the
    largest."""

    def _index(lo1: Int, hi1: Int, lo2: Int, hi2: Int) raises -> Index:
        var columns = List[ColumnZones]()
        columns.append(
            ColumnZones(
                String("k"),
                [
                    Int64Scalar(Scalar[int64.native](lo1)).to_dyn(),
                    Int64Scalar(Scalar[int64.native](lo2)).to_dyn(),
                ],
                [
                    Int64Scalar(Scalar[int64.native](hi1)).to_dyn(),
                    Int64Scalar(Scalar[int64.native](hi2)).to_dyn(),
                ],
                [0, 0],
                [50, 40],
            )
        )
        return Index([50, 40], columns^)

    var k = schema([field("k", int64)])
    var disjoint = Estimate.from_index(_index(0, 49, 50, 89), k)
    var overlapping = Estimate.from_index(_index(0, 60, 50, 89), k)
    assert_equal(disjoint.columns[0].ndv.known().value(), 90)
    assert_equal(overlapping.columns[0].ndv.known().value(), 50)
