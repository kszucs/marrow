"""ClickBench-inspired group-by benchmarks: marrow vs pyarrow, polars, duckdb.

Query definitions and per-engine runners live in ``clickbench.py``, shared with
the correctness suite ``test_clickbench.py`` — so a query is defined once and
both timed and checked. Every engine is driven through its Python API.

    pixi run -e bench pytest python/marrow/tests/bench_clickbench.py \\
        --benchmark --competition
"""

import pytest

import clickbench as cb

pytestmark = pytest.mark.skipif(
    not cb.HAVE_DATA, reason=f"ClickBench hits parquet not found at {cb.HITS}"
)


@pytest.fixture(scope="session")
def data():
    return cb.load()


@pytest.fixture(params=list(cb.QUERIES))
def qname(request):
    return request.param


def _bench(benchmark, data, engine, qname):
    if not cb.available(engine):
        pytest.skip(f"{engine} not installed")
    benchmark.extra_info.update(lib=engine, n=1_000_000, query=qname)
    q = cb.QUERIES[qname]
    benchmark(lambda: cb.run_native(engine, data, q))


@pytest.mark.benchmark(group="clickbench")
def test_marrow(benchmark, data, qname):
    _bench(benchmark, data, "marrow", qname)


@pytest.mark.benchmark(group="clickbench")
def test_pyarrow(benchmark, data, qname):
    _bench(benchmark, data, "pyarrow", qname)


@pytest.mark.benchmark(group="clickbench")
def test_polars(benchmark, data, qname):
    _bench(benchmark, data, "polars", qname)


@pytest.mark.benchmark(group="clickbench")
def test_duckdb(benchmark, data, qname):
    _bench(benchmark, data, "duckdb", qname)


@pytest.mark.benchmark(group="clickbench")
def test_datafusion(benchmark, data, qname):
    _bench(benchmark, data, "datafusion", qname)
