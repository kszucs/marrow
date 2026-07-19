"""Correctness of marrow's group-by against pyarrow, polars, duckdb (datafusion).

Runs each ClickBench-inspired query (see ``clickbench.py``) on marrow and on
every other available engine, and asserts the results agree group-for-group
(order-independent, float-tolerant). Skips cleanly when the dataset or an engine
is absent.
"""

import pytest

import clickbench as cb

pytestmark = pytest.mark.skipif(
    not cb.HAVE_DATA, reason=f"ClickBench hits parquet not found at {cb.HITS}"
)

_COMPETITORS = ["pyarrow", "polars", "duckdb", "datafusion"]


@pytest.fixture(scope="session")
def data():
    return cb.load()


def _result(engine, data, q):
    return cb.result_map(cb.to_arrow(engine, cb.run_native(engine, data, q)), q)


@pytest.mark.parametrize("qname", list(cb.QUERIES))
@pytest.mark.parametrize("engine", _COMPETITORS)
def test_marrow_matches(data, engine, qname):
    if not cb.available(engine):
        pytest.skip(f"{engine} not installed")
    q = cb.QUERIES[qname]
    marrow = _result("marrow", data, q)
    other = _result(engine, data, q)
    cb.assert_maps_equal(marrow, other)
