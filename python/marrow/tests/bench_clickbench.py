"""ClickBench-inspired group-by aggregations: marrow vs pyarrow, polars, duckdb.

Runs a handful of GROUP BY aggregations from the ClickBench workload
(https://github.com/ClickHouse/ClickBench) over the real ``hits`` dataset,
driving every engine through its Python API (apples-to-apples).

The dataset isn't vendored — point ``MARROW_CLICKBENCH_HITS`` at a ``hits``
parquet partition (e.g. ``hits_0.parquet``, 1M rows), or drop one at
``~/Workspace/ClickBench/data/hits_0.parquet``. The queries are restricted to
numeric-key group-bys with count/sum/avg, which is what marrow's ``group_by``
currently exposes.

    pixi run -e bench pytest python/marrow/tests/bench_clickbench.py \\
        --benchmark --competition
"""

import os

import pytest
import pyarrow.parquet as pq

pl = pytest.importorskip("polars")
ma = pytest.importorskip("marrow")

_DEFAULT = os.path.expanduser("~/Workspace/ClickBench/data/hits_0.parquet")
HITS = os.environ.get("MARROW_CLICKBENCH_HITS", _DEFAULT)

pytestmark = pytest.mark.skipif(
    not os.path.exists(HITS), reason=f"ClickBench hits parquet not found at {HITS}"
)

# Columns used across the queries below (a slice of the 105-column schema).
_COLS = [
    "RegionID",
    "UserID",
    "SearchEngineID",
    "AdvEngineID",
    "ResolutionWidth",
    "IsRefresh",
]

# The GROUP BY core of a spread of ClickBench queries, restricted to the
# numeric-key count/sum/avg/min/max marrow's group_by exposes today (ClickBench
# proper also uses COUNT(DISTINCT), string LIKE, and date functions marrow lacks;
# ORDER BY ... LIMIT is dropped so the comparison is the aggregation itself).
# marrow's count(<non-null key>) == COUNT(*).
QUERIES = {
    # Q7 shape — very low cardinality (~6 groups), single aggregate.
    "advengine": dict(
        key="AdvEngineID",
        aggs=[("AdvEngineID", "count")],
        pl=lambda df: df.group_by("AdvEngineID").agg(pl.len()),
        sql="SELECT AdvEngineID, count(*) FROM hits GROUP BY AdvEngineID",
    ),
    # low cardinality (~39 groups), multi-aggregate.
    "searchengine": dict(
        key="SearchEngineID",
        aggs=[
            ("SearchEngineID", "count"),
            ("IsRefresh", "sum"),
            ("ResolutionWidth", "mean"),
        ],
        pl=lambda df: df.group_by("SearchEngineID").agg(
            [pl.len(), pl.col("IsRefresh").sum(), pl.col("ResolutionWidth").mean()]
        ),
        sql="SELECT SearchEngineID, count(*), sum(IsRefresh), avg(ResolutionWidth)"
        " FROM hits GROUP BY SearchEngineID",
    ),
    # Q9 shape — mid cardinality (~1242 groups), multi-aggregate.
    "region": dict(
        key="RegionID",
        aggs=[
            ("RegionID", "count"),
            ("AdvEngineID", "sum"),
            ("ResolutionWidth", "mean"),
        ],
        pl=lambda df: df.group_by("RegionID").agg(
            [pl.len(), pl.col("AdvEngineID").sum(), pl.col("ResolutionWidth").mean()]
        ),
        sql="SELECT RegionID, count(*), sum(AdvEngineID), avg(ResolutionWidth)"
        " FROM hits GROUP BY RegionID",
    ),
    # mid cardinality, min/max aggregates.
    "region_minmax": dict(
        key="RegionID",
        aggs=[("ResolutionWidth", "min"), ("ResolutionWidth", "max")],
        pl=lambda df: df.group_by("RegionID").agg(
            [
                pl.col("ResolutionWidth").min().alias("mn"),
                pl.col("ResolutionWidth").max().alias("mx"),
            ]
        ),
        sql="SELECT RegionID, min(ResolutionWidth), max(ResolutionWidth)"
        " FROM hits GROUP BY RegionID",
    ),
    # Q14 shape — two key columns (~3045 groups).
    "multikey": dict(
        key=["RegionID", "SearchEngineID"],
        aggs=[("RegionID", "count")],
        pl=lambda df: df.group_by(["RegionID", "SearchEngineID"]).agg(pl.len()),
        sql="SELECT RegionID, SearchEngineID, count(*)"
        " FROM hits GROUP BY RegionID, SearchEngineID",
    ),
    # Q15 shape — high cardinality (~80k groups), single aggregate.
    "user": dict(
        key="UserID",
        aggs=[("UserID", "count")],
        pl=lambda df: df.group_by("UserID").agg(pl.len()),
        sql="SELECT UserID, count(*) FROM hits GROUP BY UserID",
    ),
}
_ids = list(QUERIES)


@pytest.fixture(scope="session")
def data():
    tbl = pq.read_table(HITS, columns=_COLS).combine_chunks()
    con = None
    try:
        import duckdb

        con = duckdb.connect()
        con.register("hits", tbl)
    except ImportError:
        pass
    return dict(
        tbl=tbl,
        rb=ma.record_batch(tbl.to_batches()[0]),
        df=pl.from_arrow(tbl),
        con=con,
    )


@pytest.fixture(params=_ids)
def q(request):
    return QUERIES[request.param]


@pytest.mark.benchmark(group="clickbench")
def test_marrow(benchmark, data, q, request):
    benchmark.extra_info.update(lib="marrow", query=request.node.callspec.id)
    rb, key, aggs = data["rb"], q["key"], q["aggs"]
    benchmark(lambda: rb.group_by(key).aggregate(aggs))


@pytest.mark.benchmark(group="clickbench")
def test_pyarrow(benchmark, data, q, request):
    benchmark.extra_info.update(lib="pyarrow", query=request.node.callspec.id)
    tbl, key, aggs = data["tbl"], q["key"], q["aggs"]
    benchmark(lambda: tbl.group_by(key).aggregate(aggs))


@pytest.mark.benchmark(group="clickbench")
def test_polars(benchmark, data, q, request):
    benchmark.extra_info.update(lib="polars", query=request.node.callspec.id)
    df, fn = data["df"], q["pl"]
    benchmark(lambda: fn(df))


@pytest.mark.benchmark(group="clickbench")
def test_duckdb(benchmark, data, q, request):
    if data["con"] is None:
        pytest.skip("duckdb not installed")
    benchmark.extra_info.update(lib="duckdb", query=request.node.callspec.id)
    con, sql = data["con"], q["sql"]
    benchmark(lambda: con.execute(sql).fetchall())
