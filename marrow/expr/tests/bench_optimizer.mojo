"""What a distinct count is worth, measured as the plan it changes.

`JoinReassociation` turns `(A ⋈ B) ⋈ C` into `A ⋈ (B ⋈ C)` when that is
cheaper, and "cheaper" is read off `Estimate.joined`. Until a footer's
`distinct_count` reached `ColumnEstimate.ndv`, `max_distinct` fell back to the
row count, so the containment denominator was `max(|L|, |R|)` and **every**
join estimated at `min(|L|, |R|)` rows — a many-to-many explosion predicted to
*shrink*. The rule could then only see how wide the intermediate was, never how
tall.

**Both arms run in one process against one build**, which is the point of the
file: the only difference between them is whether the three scans carry an
`ndv`, so nothing here depends on rebuilding `libmarrow.so` between
measurements or on this machine's drift between two runs.

The fixture is written by **marrow's own writer**, because pyarrow emits no
`Statistics.distinct_count` at all — `has_distinct_count` is False even for a
dictionary-encoded column — so a pyarrow fixture would leave both arms blind
and measure nothing.

    A   25,000 rows, one column `ak` over 25 distinct values, five row groups
    B    1,000 rows, `ak` over the same 25 values and `bc` unique
    C    1,000 rows, `bc` — 25 of which are B's, and 975 that match nothing

`A ⋈ B` on `ak` is many-to-many: 40 `B` rows per key, so it really produces
1,000,000 rows. `B ⋈ C` on `bc` produces 25. Both shapes answer the same
25,000 rows. Blind, the model puts `A ⋈ B` at 1,000 rows and keeps the
left-deep plan; informed, it puts it at 1,000,000 and reassociates.

`bench_reassoc_noise_a` / `_b` are the control: the same two-way join run
twice, so their spread is this run's noise floor and the arms above are read
against it rather than against a remembered number.

Run with:
    pixi run -e dev pytest marrow/expr/tests/bench_optimizer.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep

from ...builders import array
from ...dtypes import field, int64
from ...io import FileSink
from ...kernels.join import JOIN_INNER
from ...parquet.codecs import Compression
from ...parquet.reader import ParquetFile
from ...parquet.writer import FileWriter
from ...schema import Schema, schema
from ...tabular import RecordBatch, Table, record_batch
from ...utils.testing import Benchmark
from ..estimates import Approx, ColumnEstimate, Estimate
from ..index import Index
from ..logical import DynRelation, ParquetScan, ScanPath
from ..optimizer import AllRules


comptime A_ROWS = 25_000
comptime B_ROWS = 1_000
comptime C_ROWS = 1_000
comptime KEYS = 25
"""Distinct `ak` values, in both `A` and `B`. `B_ROWS // KEYS` = 40 `B` rows
share each one, which is what makes `A ⋈ B` many-to-many."""
comptime A_ROW_GROUP = 5_000
"""Five row groups over `A`, so the per-chunk distinct counts really have to be
reduced rather than read. Every group holds all 25 values, so the maximum is
exactly right — the case the maximum is chosen for."""

comptime A_PATH = "/tmp/marrow_bench_reassoc_a.parquet"
comptime B_PATH = "/tmp/marrow_bench_reassoc_b.parquet"
comptime C_PATH = "/tmp/marrow_bench_reassoc_c.parquet"


# ---------------------------------------------------------------------------
# the fixture
# ---------------------------------------------------------------------------
def _write(batch: RecordBatch, path: String, row_group: Int) raises:
    """One table through marrow's writer, dictionary-encoded by default — which
    is the only reason a `distinct_count` exists to read back."""
    var s = Schema(copy=batch.schema)
    var w = FileWriter(FileSink(path), Compression.UNCOMPRESSED)
    w.write(Table.from_batches(s^, [batch.copy()]), row_group_size=row_group)


def _fixture() raises:
    var ak = List[Optional[Int]](capacity=A_ROWS)
    for i in range(A_ROWS):
        ak.append(i % KEYS)
    _write(
        record_batch([array(ak^, int64).to_dyn()], names=["ak"]),
        String(A_PATH),
        A_ROW_GROUP,
    )

    var bak = List[Optional[Int]](capacity=B_ROWS)
    var bbc = List[Optional[Int]](capacity=B_ROWS)
    for i in range(B_ROWS):
        bak.append(i % KEYS)
        bbc.append(i)
    _write(
        record_batch(
            [array(bak^, int64).to_dyn(), array(bbc^, int64).to_dyn()],
            names=["ak", "bc"],
        ),
        String(B_PATH),
        B_ROWS,
    )

    # `bc < KEYS` are `B`'s first 25 rows, which carry 25 *distinct* `ak`, so
    # each surviving `B` row claims one whole `ak` group of `A`. The rest are
    # out of `B`'s range entirely, so they widen `C` without joining anything.
    var cbc = List[Optional[Int]](capacity=C_ROWS)
    var cv = List[Optional[Int]](capacity=C_ROWS)
    for i in range(C_ROWS):
        cbc.append(i if i < KEYS else B_ROWS + i)
        cv.append(i)
    _write(
        record_batch(
            [array(cbc^, int64).to_dyn(), array(cv^, int64).to_dyn()],
            names=["bc", "cv"],
        ),
        String(C_PATH),
        C_ROWS,
    )


def _a_schema() raises -> Schema:
    return schema([field("ak", int64)])


def _b_schema() raises -> Schema:
    return schema([field("ak", int64), field("bc", int64)])


def _c_schema() raises -> Schema:
    return schema([field("bc", int64), field("cv", int64)])


def _without_ndv(est: Estimate) raises -> Estimate:
    """`est` as this tree read it before a footer's `distinct_count` was
    carried through: every other figure intact and `ndv` unknown.

    The blind arm, spelled as a subtraction from the informed one rather than
    as a second fixture, so the two cannot differ in anything else.
    """
    var cols = List[ColumnEstimate](capacity=len(est.columns))
    for ref c in est.columns:
        cols.append(
            ColumnEstimate(
                c.name.copy(),
                c.min.copy(),
                c.max.copy(),
                nulls=c.nulls,
                ndv=Approx.unknown(),
                width=c.width,
            )
        )
    return Estimate(est.rows, cols^)


def _source(path: String, s: Schema, ndv: Bool) raises -> DynRelation:
    """One scan told what its own footer says, with or without the distinct
    counts. Reading the footer is the caller's job either way — the plan does
    no I/O until it runs."""
    var f = ParquetFile(path)
    var est = Estimate.from_index(Index.from_parquet(f), s)
    if not ndv:
        est = _without_ndv(est)
    var node = ParquetScan(ScanPath(path.copy()), s.copy()).with_statistics(
        est^
    )
    var out: DynRelation = node^
    return out^


def _three_way(ndv: Bool) raises -> DynRelation:
    """`(A ⋈ B) ⋈ C`, left-deep as written.

    The inner join is `A.ak = B.ak`; the outer is `B.bc = C.bc`, read off the
    inner join's schema `[ak, ak, bc]` at index 2. The outer predicate names a
    column of `B` and none of `A`, which is `JoinReassociation`'s condition.
    """
    var inner = _source(String(A_PATH), _a_schema(), ndv).join(
        _source(String(B_PATH), _b_schema(), ndv), [0], [0], JOIN_INNER
    )
    return inner.join(
        _source(String(C_PATH), _c_schema(), ndv), [2], [0], JOIN_INNER
    )


# ---------------------------------------------------------------------------
# the two arms
# ---------------------------------------------------------------------------
def bench_reassoc_blind(mut b: Benchmark) raises:
    """The plan chosen without a distinct count: left-deep, materialising a
    1,000,000-row intermediate the model believes is 1,000 rows."""
    _fixture()
    var plan = _three_way(ndv=False).optimize[AllRules]()
    b.throughput(BenchMetric.elements, A_ROWS)

    @always_inline
    def call() raises {imm}:
        keep(plan.execute().num_rows())

    b.iter(call)
    keep(plan)


def bench_reassoc_informed(mut b: Benchmark) raises:
    """The plan chosen with one: right-deep, building the 25 rows of
    `B ⋈ C` and probing `A` once."""
    _fixture()
    var plan = _three_way(ndv=True).optimize[AllRules]()
    b.throughput(BenchMetric.elements, A_ROWS)

    @always_inline
    def call() raises {imm}:
        keep(plan.execute().num_rows())

    b.iter(call)
    keep(plan)


# ---------------------------------------------------------------------------
# the control — identical work, twice
# ---------------------------------------------------------------------------
def _two_way() raises -> DynRelation:
    """`B ⋈ C`, which has no association to choose and so cannot move."""
    return _source(String(B_PATH), _b_schema(), ndv=True).join(
        _source(String(C_PATH), _c_schema(), ndv=True), [1], [0], JOIN_INNER
    )


def _bench_noise(mut b: Benchmark) raises:
    _fixture()
    var plan = _two_way().optimize[AllRules]()
    b.throughput(BenchMetric.elements, B_ROWS)

    @always_inline
    def call() raises {imm}:
        keep(plan.execute().num_rows())

    b.iter(call)
    keep(plan)


def bench_reassoc_noise_a(mut b: Benchmark) raises:
    """A row the change cannot touch, run twice so the pair's spread is this
    run's noise floor rather than a remembered figure."""
    _bench_noise(b)


def bench_reassoc_noise_b(mut b: Benchmark) raises:
    """The other half of the control pair — the same work as
    `bench_reassoc_noise_a`, so any difference between them is drift."""
    _bench_noise(b)
