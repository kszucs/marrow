"""The index: what a source knows about its data, and the mask a predicate
reads off it.

Three kinds of case, and the last is the one that matters.

**The index itself** — one statistic per chunk, in the type the reading node
asks for. Each case pins a rule where a plausible implementation is silently
wrong rather than obviously wrong: an unrecorded column reads null rather than
absent, a missing statistic blinds its own chunk and no other, an unrecorded
null count is not zero, and an all-null chunk is skipped on the counts alone
with no bounds at all.

**Both lanes reach the same answer.** They share no node types and get at the
statistics differently — a fused node passes its own `T`, an interpreted one
recovers the dtype from the index and dispatches — so several cases assert the
two masks are equal rather than merely each correct. A predicate that pruned
differently depending on how it was spelled would be invisible otherwise, and
the runtime lane is the only one that prunes a temporal column at all.

**`test_index_never_excludes_a_chunk_that_holds_a_match`** then brute-forces
the soundness property: for every interval and every threshold in a small
range, a chunk that is dropped must contain no value satisfying the predicate.
A wrong "keep" costs time; a wrong "drop" is a wrong answer, so the sweep
asserts the implication in the only direction that matters — and a mask that
answered all-true would pass it, which is why the table-driven cases above
assert exact bit patterns.
"""

from std.testing import assert_equal, assert_false, assert_true

from ...arrays import BoolArray
from ...builders import array
from ...dtypes import Date32Type, Int64Type, date32, int64
from ...scalars import Date32Scalar, DynScalar, Int64Scalar
from ..bindings import Bindings
from ..builders import col, lit, param
from ..index import ColumnZones, Index, ZoneMaps
from ...parquet.reader import PageBounds
from ..logical import DynValue
from ..runtime.values import and_, column, eq, gt, literal


# ---------------------------------------------------------------------------
# The index -- what a source knows before reading
# ---------------------------------------------------------------------------
def _index(var cols: List[ColumnZones], chunks: Int) raises -> Index:
    """An index over `chunks` chunks, described by these columns."""
    var z = ZoneMaps(capacity=len(cols))
    for ref c in cols:
        z.add(c.copy())
    return Index(chunks=chunks, zones=z^)


def _i64(v: Int) -> DynScalar:
    return Int64Scalar(Scalar[int64.native](v)).to_dyn()


def _date(v: Int) -> DynScalar:
    """A `date32`, which needs its dtype spelled: a temporal type carries a
    unit and so is not `Defaultable`, the same reason `ZoneMaps._stats` takes
    the witness its caller already holds."""
    return Date32Scalar(
        Optional(Scalar[Date32Type.native](v)), date32()
    ).to_dyn()


def _absent() -> DynScalar:
    """A statistic the source did not record."""
    return Int64Scalar(Optional[Scalar[int64.native]](None)).to_dyn()


def test_index_reads_one_statistic_for_every_chunk() raises:
    """Three chunks in, three-element arrays out, in the node's own type."""
    var idx = _index(
        [
            ColumnZones(
                String("a"),
                [_i64(0), _i64(10), _i64(20)],
                [_i64(9), _i64(19), _i64(29)],
                [0, 0, 0],
            )
        ],
        chunks=3,
    )
    assert_true(
        idx.mins[Int64Type](String("a"), int64) == array([0, 10, 20], int64)
    )
    assert_true(
        idx.maxes[Int64Type](String("a"), int64) == array([9, 19, 29], int64)
    )


def test_index_an_unrecorded_column_answers_all_null() raises:
    """A column the source never wrote statistics for prunes nothing.

    Null rather than absent, because a null propagates through the comparison
    and a null answer already means "cannot prove, read it" -- the same rule an
    unrecognised predicate gets, expressed once in the data.
    """
    var mins = Index(chunks=2).mins[Int64Type](String("missing"), int64)
    assert_equal(len(mins), 2)
    assert_equal(mins.null_count(), 2)


def test_index_a_missing_statistic_is_null_for_that_chunk_alone() raises:
    """One chunk without bounds does not blind the others."""
    var idx = _index(
        [
            ColumnZones(
                String("a"),
                [_i64(0), _absent(), _i64(20)],
                [_i64(9), _absent(), _i64(29)],
                [0, -1, 0],
            )
        ],
        chunks=3,
    )
    var maxes = idx.maxes[Int64Type](String("a"), int64)
    assert_equal(maxes.null_count(), 1)
    assert_true(maxes.is_valid(0) and maxes.is_null(1) and maxes.is_valid(2))


def test_index_an_unrecorded_null_count_is_not_zero() raises:
    """`-1` means the source stored none. Reading it as zero would be a
    soundness choice this type declines to make."""
    var idx = _index(
        [
            ColumnZones(
                String("a"), [_i64(0), _i64(1)], [_i64(9), _i64(9)], [-1, 4]
            )
        ],
        chunks=2,
    )
    assert_equal(idx.zones.null_counts(String("a")), [-1, 4])


def test_index_an_all_null_chunk_is_skipped_without_any_bounds() raises:
    """The one exactly-provable skip, and the one that needs no statistics.

    A chunk whose column is entirely null yields no surviving row for *any*
    comparison — every comparison with NULL is NULL, and a filter keeps only
    valid true bits. Chunk 1 below records no bounds at all and is still
    dropped, on the null count and the row count alone.
    """
    var idx = _index(
        [
            ColumnZones(
                String("a"),
                [_i64(0), _absent(), _i64(20)],
                [_i64(9), _absent(), _i64(29)],
                [0, 5, 0],
            )
        ],
        chunks=3,
    )
    idx.rows = [5, 5, 5]
    assert_equal(_bits(idx.defined(String("a"))), [1, 0, 1])
    assert_equal(_bits((col("a", int64) > lit(-1, int64)).mask(idx)), [1, 0, 1])


def test_index_an_unknown_row_count_cannot_prove_all_null() raises:
    """`defined` needs both numbers. A null count with no row count to compare
    it against proves nothing, so the chunk is kept."""
    var idx = _index(
        [ColumnZones(String("a"), [_absent()], [_absent()], [5])], chunks=1
    )
    assert_equal(_bits(idx.defined(String("a"))), [1])


def test_index_an_empty_index_knows_nothing() raises:
    """The default. Every lookup is all-null, so nothing prunes."""
    var idx = Index()
    assert_equal(idx.chunks, 0)
    assert_equal(idx.zones.num_columns(), 0)


# ---------------------------------------------------------------------------
# mask -- the predicate read over the index
# ---------------------------------------------------------------------------
def _three_chunks() raises -> Index:
    """Three chunks holding a in [0,9], [10,19], [20,29]."""
    return _index(
        [
            ColumnZones(
                String("a"),
                [_i64(0), _i64(10), _i64(20)],
                [_i64(9), _i64(19), _i64(29)],
                [0, 0, 0],
            )
        ],
        chunks=3,
    )


def _bits(mask: BoolArray) -> List[Int]:
    """The mask as 1 / 0 / -1 for true / false / null."""
    var out = List[Int](capacity=len(mask))
    for i in range(len(mask)):
        if mask.is_null(i):
            out.append(-1)
        else:
            out.append(1 if mask[i].value() else 0)
    return out^


def test_index_greater_reads_the_upper_extreme() raises:
    """`a > 15` can only hold where some value exceeds 15 — chunks 1 and 2."""
    var m = (col("a", int64) > lit(15, int64)).mask(_three_chunks())
    assert_equal(_bits(m), [0, 1, 1])


def test_index_less_reads_the_lower_extreme() raises:
    var m = (col("a", int64) < lit(15, int64)).mask(_three_chunks())
    assert_equal(_bits(m), [1, 1, 0])


def test_index_conjunction_intersects_the_masks() raises:
    """`a > 15 AND a < 25` keeps only the chunk that can satisfy both."""
    var p = (col("a", int64) > lit(15, int64)) & (
        col("a", int64) < lit(25, int64)
    )
    assert_equal(_bits(p.mask(_three_chunks())), [0, 1, 1])


def test_index_disjunction_keeps_what_either_side_keeps() raises:
    var p = (col("a", int64) < lit(5, int64)) | (
        col("a", int64) > lit(25, int64)
    )
    assert_equal(_bits(p.mask(_three_chunks())), [1, 0, 1])


def test_index_an_unknown_operand_keeps_every_chunk() raises:
    """A column the index never recorded proves nothing, so nothing is
    skipped. The mask is null rather than false — the distinction that keeps a
    missing statistic from being read as a proof."""
    var m = (col("zz", int64) > lit(15, int64)).mask(_three_chunks())
    assert_equal(_bits(m), [-1, -1, -1])


def test_index_an_empty_index_prunes_nothing() raises:
    """A source that knows nothing about itself keeps everything it has."""
    var m = (col("a", int64) > lit(15, int64)).mask(Index(chunks=2))
    assert_equal(_bits(m), [-1, -1])


def test_index_equality_is_interval_overlap() raises:
    """`a = 15` can only hold where 15 falls inside the chunk's range."""
    var m = (col("a", int64) == lit(15, int64)).mask(_three_chunks())
    assert_equal(_bits(m), [0, 1, 0])


def test_index_a_parameter_prunes_with_this_execution_s_value() raises:
    """`a > :threshold` prunes, and prunes *differently* per execution.

    The AOT lane's whole surface is a plan with a late-bound threshold, so a
    parameter that said nothing would leave every such query reading the whole
    file. The value is read from `Bindings` at the same point a literal is read
    off the node — which is why `mask` carries them.
    """
    var p = col("a", int64) > param("t", int64)
    var low: Bindings = {"t": _i64(15)}
    var high: Bindings = {"t": _i64(25)}
    assert_equal(_bits(p.mask(_three_chunks(), low)), [0, 1, 1])
    assert_equal(_bits(p.mask(_three_chunks(), high)), [0, 0, 1])


# ---------------------------------------------------------------------------
# the runtime lane -- the same contract, reached by recovering the dtype
# ---------------------------------------------------------------------------
def test_index_the_runtime_lane_reads_the_same_extremes() raises:
    """`RuntimeValue.mask` answers what the fused node answers.

    The two lanes share no node types and reach the statistics differently — a
    comptime node passes its own `T`, an interpreted one recovers the dtype
    from the index and dispatches. The answers have to agree, or a plan prunes
    differently depending on how it was spelled.
    """
    var runtime = gt(column("a"), literal(_i64(15)))
    var fused = col("a", int64) > lit(15, int64)
    assert_equal(_bits(runtime.mask(_three_chunks())), [0, 1, 1])
    assert_equal(
        _bits(runtime.mask(_three_chunks())),
        _bits(fused.mask(_three_chunks())),
    )


def test_index_the_runtime_lane_composes_under_and() raises:
    """Both lanes fold with the same Kleene kernel, so a conjunction narrows
    the same way — asserted against the fused spelling rather than a
    hand-computed answer, which is how this case first got its own expectation
    wrong."""
    var runtime = and_(
        gt(column("a"), literal(_i64(15))),
        gt(literal(_i64(25)), column("a")),
    )
    var fused = (col("a", int64) > lit(15, int64)) & (
        lit(25, int64) > col("a", int64)
    )
    assert_equal(_bits(runtime.mask(_three_chunks())), [0, 1, 1])
    assert_equal(
        _bits(runtime.mask(_three_chunks())),
        _bits(fused.mask(_three_chunks())),
    )


def _dates() raises -> Index:
    """Three chunks holding d in [0,99], [100,199], [200,299], as `date32`."""
    return _index(
        [
            ColumnZones(
                String("d"),
                [_date(0), _date(100), _date(200)],
                [_date(99), _date(199), _date(299)],
                [0, 0, 0],
            )
        ],
        chunks=3,
    )


def test_index_the_runtime_lane_prunes_a_date_column() raises:
    """The lane that makes the temporal claim true.

    A `date32` predicate prunes here and nowhere else: the comparison kernels
    take every `PrimitiveType`, and `RuntimeValue._statistics` recovers the
    dtype from the statistics themselves. The comptime lane cannot express
    `date_col > date_const` at all — `lit` has no temporal overload — so
    `TemporalCompare` keeps every chunk, correctly and uselessly.
    """
    var idx = _dates()
    var p = gt(column("d"), literal(_date(150)))
    assert_equal(_bits(p.mask(idx)), [0, 1, 1])


def test_index_the_runtime_lane_keeps_a_column_it_has_no_statistics_for() raises:
    """An unrecorded column proves nothing, and says so *locally*.

    The erased kernel entry raises on a dtype mismatch rather than answering
    null, and `Index.read_plan` has one `except` covering every predicate — so
    without a guard here, one predicate naming a column the source never
    recorded would silently cost every other predicate on that scan its
    pruning. The mask is all-true rather than all-null for the same reason the
    trait default is.
    """
    var p = gt(column("zz"), literal(_i64(15)))
    assert_equal(_bits(p.mask(_three_chunks())), [1, 1, 1])


def test_index_one_blind_predicate_does_not_disable_the_others() raises:
    """The consequence, at the level that matters: a read plan built from a
    working predicate and a blind one still skips what the working one
    proves."""
    var idx = _three_chunks()
    var predicates: List[DynValue] = [
        DynValue(gt(column("a"), literal(_i64(15)))),
        DynValue(gt(column("zz"), literal(_i64(15)))),
    ]
    assert_equal(idx.read_plan(predicates), [1, 2])


# ---------------------------------------------------------------------------
# pages -- the same index one granularity down
# ---------------------------------------------------------------------------
def _page(rows: Int, lo: Int, hi: Int) -> PageBounds:
    return PageBounds(rows, Optional(_i64(lo)), Optional(_i64(hi)))


def _null_page(rows: Int) -> PageBounds:
    """A page the writer marked all-null: no bounds at all."""
    return PageBounds(rows, None, None)


def test_index_pages_are_chunks_like_any_other() raises:
    """`Index.from_pages` builds the same thing `from_parquet` does, one
    granularity down — so `mask` and `surviving` need to know nothing about
    pages."""
    var idx = Index.from_pages(
        String("a"), [_page(10, 0, 9), _page(10, 10, 19), _page(10, 20, 29)], 30
    )
    assert_equal(idx.chunks, 3)
    assert_equal(idx.rows, [10, 10, 10])
    assert_equal(_bits((col("a", int64) > lit(15, int64)).mask(idx)), [0, 1, 1])


def test_index_a_page_with_no_bounds_is_never_skipped() raises:
    """The page-level half of "cannot prove, read it".

    `page_bounds` records no min/max for a page in `ColumnIndex.null_pages`,
    and a page index carries no null *count* either — so an all-null page is
    exactly the chunk nothing can be proven about. Skipping it would happen to
    be sound, since a null satisfies no comparison, but that is an argument
    about the predicate and this layer does not make it: it answers from the
    bounds, and it has none.
    """
    var idx = Index.from_pages(
        String("a"), [_page(10, 0, 9), _null_page(10), _page(10, 20, 29)], 30
    )
    assert_equal(
        _bits((col("a", int64) > lit(15, int64)).mask(idx)), [0, -1, 1]
    )
    assert_equal(
        idx.surviving([DynValue(col("a", int64) > lit(15, int64))]),
        [False, True, True],
    )


def test_index_a_page_index_that_misses_rows_is_refused() raises:
    """Pages that do not tile the whole row group are a page index this cannot
    trust — answered as *no chunks*, which the caller reads as "this column
    says nothing" rather than as a selection that would misalign every column
    after it."""
    var short = Index.from_pages(String("a"), [_page(10, 0, 9)], 30)
    assert_equal(short.chunks, 0)
    var none = Index.from_pages(String("a"), List[PageBounds](), 30)
    assert_equal(none.chunks, 0)


def test_index_never_excludes_a_chunk_that_holds_a_match() raises:
    """The soundness property, swept exhaustively.

    For every interval and every threshold in a small range, a chunk that is
    dropped must contain no value satisfying the predicate. A wrong `keep`
    costs time; a wrong `drop` is a wrong answer, so this asserts the
    implication in the only direction that matters.
    """
    for lo in range(-3, 4):
        for hi in range(lo, 4):
            for t in range(-4, 5):
                var idx = _index(
                    [ColumnZones(String("a"), [_i64(lo)], [_i64(hi)], [0])],
                    chunks=1,
                )
                idx.rows = [1]

                var gt = (col("a", int64) > lit(t, int64)).mask(idx)
                if not gt.is_null(0) and not gt[0].value():
                    for v in range(lo, hi + 1):
                        assert_true(
                            not (v > t),
                            "dropped a chunk holding a match for >",
                        )

                var eq = (col("a", int64) == lit(t, int64)).mask(idx)
                if not eq.is_null(0) and not eq[0].value():
                    for v in range(lo, hi + 1):
                        assert_true(
                            v != t,
                            "dropped a chunk holding a match for ==",
                        )
