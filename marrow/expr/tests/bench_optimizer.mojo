# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""What the join search is worth: plans it changes, and what planning costs.

**Distinct counts decide the plan.** `(A ⋈ B) ⋈ C` over three Parquet files,
once with each footer's `distinct_count` and once without. Blind, the model
puts `A ⋈ B` at 1,000 rows — `max_distinct` falls back to the row count, the
containment denominator is `max(|L|, |R|)`, and a many-to-many explosion is
predicted to *shrink* — and keeps the written order; informed, it puts it at
1,000,000 and joins `B ⋈ C`, 25 rows, first. Both arms run in one process
against one build, so nothing depends on rebuilding between measurements.

The fixture is written by **marrow's own writer**, because pyarrow emits no
`Statistics.distinct_count` at all.

    A   25,000 rows, one column `ak` over 25 distinct values, five row groups
    B    1,000 rows, `bk` over the same 25 values and `bc` unique
    C    1,000 rows, `cc` — 25 of which are B's `bc`, and 975 that match nothing

`bench_reassoc_noise_a` / `_b` are the control: the same two-way join run
twice, so their spread is this run's noise floor.

**Written against optimized.** A star and a chain over in-memory tables
carrying the statistics `analyze` would find, each written in an order the
search beats, run as written and as the search orders them.

**Planning.** `optimize[AllRules]()` alone over scans that carry statistics and
are never opened: DPccp over a 10-dimension star, a 12-chain, an 8-cycle and a
13-leaf snowflake, and the greedy fallback past the pair budget over a
24-dimension star and a 60-chain.

Run with:
    pixi run -e dev pytest marrow/expr/tests/bench_optimizer.mojo --benchmark
"""

from std.benchmark import BenchMetric, keep
from std.os.path import join

from ...arrays import DynArray
from ...builders import array
from ...dtypes import Field, field, int64
from ...io import FileSink
from ...kernels.join import JOIN_INNER
from ...scalars import Int64Scalar
from ...parquet.codecs import Compression
from ...parquet.reader import ParquetFile
from ...parquet.writer import FileWriter
from ...schema import Schema, schema
from ...tabular import RecordBatch, Table, record_batch
from ...utils.testing import Benchmark, ScratchDir
from ..estimates import Approx, ColumnEstimate, Estimate
from ..index import Index
from ..builders import col, lit, table
from ..logical import (
    DynRelation,
    InMemoryTable,
    JoinChain,
    ParquetScan,
    ScanPath,
)
from ..optimizer import AllRules, JoinOrdering


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

comptime A_FILE = "marrow_bench_reassoc_a.parquet"
comptime B_FILE = "marrow_bench_reassoc_b.parquet"
comptime C_FILE = "marrow_bench_reassoc_c.parquet"


# ---------------------------------------------------------------------------
# the fixture
# ---------------------------------------------------------------------------
def _write(batch: RecordBatch, path: String, row_group: Int) raises:
    """One table through marrow's writer, dictionary-encoded by default — which
    is the only reason a `distinct_count` exists to read back."""
    var s = Schema(copy=batch.schema)
    var w = FileWriter(FileSink(path), Compression.UNCOMPRESSED)
    w.write(Table.from_batches(s^, [batch.copy()]), row_group_size=row_group)


def _fixture(dir: String) raises:
    var ak = List[Optional[Int]](capacity=A_ROWS)
    for i in range(A_ROWS):
        ak.append(i % KEYS)
    _write(
        record_batch([array(ak^, int64).to_dyn()], names=["ak"]),
        join(dir, A_FILE),
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
            names=["bk", "bc"],
        ),
        join(dir, B_FILE),
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
            names=["cc", "cv"],
        ),
        join(dir, C_FILE),
        C_ROWS,
    )


def _a_schema() raises -> Schema:
    return schema([field("ak", int64)])


def _b_schema() raises -> Schema:
    return schema([field("bk", int64), field("bc", int64)])


def _c_schema() raises -> Schema:
    return schema([field("cc", int64), field("cv", int64)])


def _without_ndv(est: Estimate) raises -> Estimate:
    """`est` with nothing known about how many values its columns hold:
    `ndv` unknown and the bounds dropped, since `span` reads a count off
    integer bounds. The blind arm, spelled as a subtraction from the informed
    one, so the two cannot differ in anything else."""
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


def _three_way(dir: String, ndv: Bool) raises -> DynRelation:
    """`(A ⋈ B) ⋈ C`, left-deep as written: `A.ak = B.bk`, then
    `B.bc = C.cc`, read off the inner join's schema `[ak, bk, bc]` at
    index 2."""
    var inner = _source(join(dir, A_FILE), _a_schema(), ndv).join(
        _source(join(dir, B_FILE), _b_schema(), ndv), [0], [0], JOIN_INNER
    )
    return inner.join(
        _source(join(dir, C_FILE), _c_schema(), ndv), [2], [0], JOIN_INNER
    )


# ---------------------------------------------------------------------------
# the two arms
# ---------------------------------------------------------------------------
def bench_reassoc_blind(mut b: Benchmark) raises:
    """The plan chosen without a distinct count: as written, materialising a
    1,000,000-row intermediate the model believes is 1,000 rows."""
    with ScratchDir() as dir:
        _fixture(dir)
        _bench_execute(b, _three_way(dir, ndv=False).optimize[AllRules]())


def bench_reassoc_informed(mut b: Benchmark) raises:
    """The plan chosen with one: `B ⋈ C`, 25 rows, first, and `A` probed
    once."""
    with ScratchDir() as dir:
        _fixture(dir)
        _bench_execute(b, _three_way(dir, ndv=True).optimize[AllRules]())


# ---------------------------------------------------------------------------
# the control — identical work, twice
# ---------------------------------------------------------------------------
def _two_way(dir: String) raises -> DynRelation:
    """`B ⋈ C`, which has no association to choose and so cannot move."""
    return _source(join(dir, B_FILE), _b_schema(), ndv=True).join(
        _source(join(dir, C_FILE), _c_schema(), ndv=True), [1], [0], JOIN_INNER
    )


def _bench_noise(mut b: Benchmark) raises:
    with ScratchDir() as dir:
        _fixture(dir)
        _bench_execute(b, _two_way(dir).optimize[AllRules]())


def bench_reassoc_noise_a(mut b: Benchmark) raises:
    """A row the change cannot touch, run twice so the pair's spread is this
    run's noise floor rather than a remembered figure."""
    _bench_noise(b)


def bench_reassoc_noise_b(mut b: Benchmark) raises:
    """The other half of the control pair — the same work as
    `bench_reassoc_noise_a`, so any difference between them is drift."""
    _bench_noise(b)


# ---------------------------------------------------------------------------
# written against optimized, over analysed in-memory tables
# ---------------------------------------------------------------------------
def _keyed(
    names: List[String], rows: Int, modulus: List[Int]
) raises -> DynRelation:
    """A table of `rows` rows, column `j` holding `i % modulus[j]`, carrying
    the statistics `analyze` would find — written out from the generator
    rather than computed, so this unit does not instantiate `analyze`'s
    per-dtype kernels at `-O3`."""
    var columns = List[DynArray](capacity=len(names))
    var summaries = List[ColumnEstimate](capacity=len(names))
    for j in range(len(names)):
        var values = List[Optional[Int]](capacity=rows)
        for i in range(rows):
            values.append(i % modulus[j])
        columns.append(array(values^, int64).to_dyn())
        var distinct = min(rows, modulus[j])
        summaries.append(
            ColumnEstimate(
                names[j].copy(),
                Int64Scalar(Scalar[int64.native](0)).to_dyn(),
                Int64Scalar(Scalar[int64.native](distinct - 1)).to_dyn(),
                nulls=Approx.exact(0),
                ndv=Approx.estimated(distinct),
                width=Approx.exact(8),
            )
        )
    var node = InMemoryTable(
        record_batch(columns^, names=names.copy())
    ).with_statistics(Estimate(Approx.exact(rows), summaries^))
    var out: DynRelation = node^
    return out^


def _star() raises -> DynRelation:
    """200,000 facts against three dimensions, the largest joined first and
    the one a filter shrinks to two rows joined last."""
    var fact = _keyed(["f1", "f2", "f3"], 200_000, [5_000, 100, 50])
    return (
        fact.join(_keyed(["d1"], 5_000, [5_000]), [0], [0], JOIN_INNER)
        .join(_keyed(["d2"], 100, [100]), [1], [0], JOIN_INNER)
        .join(
            _keyed(["d3"], 50, [50]).filter(col("d3", int64) < lit(2, int64)),
            [2],
            [0],
            JOIN_INNER,
        )
    )


def _chain() raises -> DynRelation:
    """`A - B - C - D`, where joining `C` before `D` fans out a thousandfold."""
    return (
        _keyed(["ax"], 10_000, [500])
        .join(_keyed(["bx", "by"], 2_000, [500, 20]), [0], [0], JOIN_INNER)
        .join(_keyed(["cy", "cz"], 20_000, [20, 10_000]), [2], [0], JOIN_INNER)
        .join(_keyed(["dz"], 10, [10_000]), [4], [0], JOIN_INNER)
    )


def _bench_execute(mut b: Benchmark, plan: DynRelation) raises:
    var rows = plan.execute().num_rows()
    b.throughput(BenchMetric.elements, rows)

    @always_inline
    def call() raises {imm}:
        keep(plan.execute().num_rows())

    b.iter(call)
    keep(plan)


def bench_join_order_star_written(mut b: Benchmark) raises:
    _bench_execute(b, _star())


def bench_join_order_star_optimized(mut b: Benchmark) raises:
    _bench_execute(b, _star().optimize[AllRules]())


def bench_join_order_chain_written(mut b: Benchmark) raises:
    _bench_execute(b, _chain())


def bench_join_order_chain_optimized(mut b: Benchmark) raises:
    _bench_execute(b, _chain().optimize[AllRules]())


# ---------------------------------------------------------------------------
# planning time — scans that carry statistics and are never opened
# ---------------------------------------------------------------------------
def _stats_scan(
    names: List[String], rows: Int, distinct: List[Int]
) raises -> DynRelation:
    """A scan whose footer says `rows` rows and, per column, `distinct[j]`
    values in `[0, distinct[j])`. Nothing opens the file."""
    var fields = List[Field](capacity=len(names))
    var cols = List[ColumnEstimate](capacity=len(names))
    for j in range(len(names)):
        fields.append(field(names[j].copy(), int64))
        cols.append(
            ColumnEstimate(
                names[j].copy(),
                Int64Scalar(Scalar[int64.native](0)).to_dyn(),
                Int64Scalar(Scalar[int64.native](distinct[j] - 1)).to_dyn(),
                nulls=Approx.exact(0),
                ndv=Approx.estimated(distinct[j]),
                width=Approx.exact(8),
            )
        )
    var node = ParquetScan(
        ScanPath(String("/nonexistent.parquet")), schema(fields^)
    ).with_statistics(Estimate(Approx.exact(rows), cols^))
    var out: DynRelation = node^
    return out^


def _star_of(dimensions: Int) raises -> DynRelation:
    """A fact with one key per dimension, the dimensions joined largest
    first."""
    var names = List[String]()
    var distinct = List[Int]()
    for d in range(dimensions):
        names.append(String("f", d))
        distinct.append(10 * (dimensions - d))
    var plan = _stats_scan(names, 1_000_000, distinct)
    for d in range(dimensions):
        plan = plan.join(
            _stats_scan([String("d", d)], distinct[d], [distinct[d]]),
            [d],
            [0],
            JOIN_INNER,
        )
    return plan^


def _chain_of(n: Int) raises -> DynRelation:
    """`n` scans, each keyed to the next, alternating large and small."""
    var plan = _stats_scan(["c0_r"], 1_000, [100])
    for i in range(1, n):
        var rows = 100_000 if i % 2 == 0 else 1_000
        plan = plan.join(
            _stats_scan(
                [String("c", i, "_l"), String("c", i, "_r")],
                rows,
                [100, 100],
            ),
            [len(plan.schema().fields) - 1],
            [0],
            JOIN_INNER,
        )
    return plan^


def _cycle_of(n: Int) raises -> DynRelation:
    """A chain of `n` whose last join also closes the cycle to the first."""
    var plan = _stats_scan(["c0_r", "c0_l"], 1_000, [100, 100])
    for i in range(1, n):
        var rows = 50_000 if i % 3 == 0 else 2_000
        var keys_l: List[Int] = [len(plan.schema().fields) - 1]
        var keys_r: List[Int] = [0]
        if i == n - 1:
            keys_l.append(1)
            keys_r.append(1)
        plan = plan.join(
            _stats_scan(
                [
                    String("c", i, "_l"),
                    String("c", i, "_z"),
                    String("c", i, "_r"),
                ],
                rows,
                [100, 100, 100],
            ),
            keys_l^,
            keys_r^,
            JOIN_INNER,
        )
    return plan^


def _snowflake() raises -> DynRelation:
    """A fact, four dimensions, and two sub-dimensions under each."""
    var plan = _stats_scan(
        ["f0", "f1", "f2", "f3"], 1_000_000, [1000, 500, 200, 100]
    )
    for d in range(4):
        var width = len(plan.schema().fields)
        plan = plan.join(
            _stats_scan(
                [String("d", d), String("d", d, "a"), String("d", d, "b")],
                1000 // (d + 1),
                [1000 // (d + 1), 50, 20],
            ),
            [d],
            [0],
            JOIN_INNER,
        )
        for s in range(2):
            plan = plan.join(
                _stats_scan(
                    [String("s", d, "_", s)],
                    50 if s == 0 else 20,
                    [50 if s == 0 else 20],
                ),
                [width + 1 + s],
                [0],
                JOIN_INNER,
            )
    return plan^


def _bench_plan(mut b: Benchmark, plan: DynRelation) raises:
    """What choosing `plan`'s join tree takes: `JoinOrdering`'s search."""
    b.throughput(BenchMetric.elements, 1)

    @always_inline
    def call() raises {imm}:
        keep(JoinOrdering.order(plan.get[JoinChain]()))

    b.iter(call)
    keep(plan)


def bench_join_order_plan_star_10(mut b: Benchmark) raises:
    _bench_plan(b, _star_of(10))


def bench_join_order_plan_chain_12(mut b: Benchmark) raises:
    _bench_plan(b, _chain_of(12))


def bench_join_order_plan_cycle_8(mut b: Benchmark) raises:
    _bench_plan(b, _cycle_of(8))


def bench_join_order_plan_snowflake_13(mut b: Benchmark) raises:
    _bench_plan(b, _snowflake())


def bench_join_order_plan_star_24_greedy(mut b: Benchmark) raises:
    _bench_plan(b, _star_of(24))


def bench_join_order_plan_chain_60_greedy(mut b: Benchmark) raises:
    _bench_plan(b, _chain_of(60))
