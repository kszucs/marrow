"""Tests for the lazy relational frontend (`marrow.LazyTable`)."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import marrow
from marrow.expr import _HAVE_EXPRESSIONS

needs_expressions = pytest.mark.skipif(
    not _HAVE_EXPRESSIONS,
    reason="marrow._expr_column (the expression bindings) is not built in",
)


@pytest.fixture
def batch():
    return marrow.record_batch(
        {
            "k": marrow.array(["a", "b", "a", "b", "a", "c"]).unwrap(),
            "v": marrow.array([1, 2, 3, 4, 5, 6]).unwrap(),
        }
    )


@pytest.fixture
def lazy(batch):
    return marrow.scan(batch)


# ── plan construction ──────────────────────────────────────────────────────


def test_scan_is_lazy(lazy):
    assert isinstance(lazy, marrow.LazyTable)
    assert lazy.column_names == ["k", "v"]
    assert "InMemoryTable" in str(lazy)


def test_repr_shows_the_plan(lazy):
    assert "LazyTable" in repr(lazy)
    assert "InMemoryTable" in repr(lazy)


def test_select(lazy):
    assert lazy.select("k").column_names == ["k"]
    assert lazy.select("v", "k").column_names == ["v", "k"]
    assert lazy.select(["k", "v"]).column_names == ["k", "v"]


def test_drop(lazy):
    assert lazy.drop("k").column_names == ["v"]
    with pytest.raises(KeyError):
        lazy.drop("nope")


def test_rename(lazy):
    assert lazy.rename({"v": "value"}).column_names == ["k", "value"]


def test_select_unknown_column_raises(lazy):
    with pytest.raises(Exception):
        lazy.select("nope").collect()


# ── execution ──────────────────────────────────────────────────────────────


def test_collect_returns_a_record_batch(lazy):
    out = lazy.collect()
    assert isinstance(out, marrow.RecordBatch)
    assert out.num_rows() == 6


def test_end_to_end_aggregate_sort_limit(lazy):
    """batch -> aggregate -> sort -> limit -> collect."""
    result = (
        lazy.aggregate(by=["k"], total=("sum", "v"), n=("count", "v"))
        .order_by(("total", "descending"))
        .limit(2)
        .to_pyarrow()
    )
    assert result.column_names == ["k", "total", "n"]
    assert result.column("k").to_pylist() == ["a", "b"]
    assert result.column("total").to_pylist() == [9, 6]
    assert result.column("n").to_pylist() == [3, 2]


def test_matches_pyarrow(batch, lazy):
    """Cross-check the grouped sum against PyArrow on the same data."""
    got = lazy.aggregate(by=["k"], total=("sum", "v")).order_by("k").to_pyarrow()

    expected = (
        pa.table(pa.record_batch(batch))
        .group_by("k")
        .aggregate([("v", "sum")])
        .sort_by("k")
    )

    assert got.column("k").to_pylist() == expected.column("k").to_pylist()
    assert got.column("total").to_pylist() == expected.column("v_sum").to_pylist()


def test_aggregate_with_no_keys(lazy):
    out = lazy.aggregate(by=[], total=("sum", "v")).to_pyarrow()
    assert out.num_rows == 1
    assert out.column("total").to_pylist() == [21]


def test_aggregate_needs_an_aggregate(lazy):
    with pytest.raises(ValueError):
        lazy.aggregate(by=["k"])


def test_order_by_ascending_is_the_default(lazy):
    out = lazy.aggregate(by=["k"], total=("sum", "v")).order_by("total")
    assert out.to_pyarrow().column("total").to_pylist() == [6, 6, 9]


def test_head_and_offset(lazy):
    assert lazy.order_by("v").head(3).to_pyarrow().column("v").to_pylist() == [
        1,
        2,
        3,
    ]
    assert lazy.order_by("v").limit(2, 2).to_pyarrow().column("v").to_pylist() == [3, 4]


def test_plan_is_reusable(lazy):
    """A plan is a description, so executing it twice gives the same rows."""
    plan = lazy.aggregate(by=["k"], total=("sum", "v")).order_by("k")
    assert plan.to_pyarrow() == plan.to_pyarrow()


def test_join(batch):
    right = marrow.record_batch(
        {
            "k": marrow.array(["a", "b"]).unwrap(),
            "label": marrow.array(["Alpha", "Beta"]).unwrap(),
        }
    )
    out = marrow.scan(batch).join(marrow.scan(right), on="k").order_by("v").to_pyarrow()
    assert out.column("label").to_pylist() == [
        "Alpha",
        "Beta",
        "Alpha",
        "Beta",
        "Alpha",
    ]


def test_join_needs_keys(batch):
    with pytest.raises(ValueError):
        marrow.scan(batch).join(marrow.scan(batch))


# ── parquet ────────────────────────────────────────────────────────────────


def test_read_parquet_infers_the_schema(tmp_path):
    path = tmp_path / "t.parquet"
    pq.write_table(pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]}), path)

    t = marrow.read_parquet(path)
    assert t.column_names == ["a", "b"]
    assert "ParquetScan" in str(t)
    assert t.to_pyarrow().column("a").to_pylist() == [1, 2, 3]


def test_read_parquet_covers_every_row_group(tmp_path):
    """`collect` drains the whole plan, not just the first row group."""
    path = tmp_path / "many.parquet"
    n = 1000
    pq.write_table(pa.table({"a": list(range(n))}), path, row_group_size=100)
    assert pq.ParquetFile(path).num_row_groups == 10

    t = marrow.read_parquet(path)
    assert t.collect().num_rows() == n
    assert t.aggregate(by=[], total=("sum", "a")).to_pyarrow().column(
        "total"
    ).to_pylist() == [sum(range(n))]


# ── expression-dependent surface ───────────────────────────────────────────


@needs_expressions
def test_getitem_returns_a_column(lazy):
    assert lazy["v"] is not None


@needs_expressions
def test_filter(lazy):
    out = lazy.filter(lazy["v"] > 3).order_by("v").to_pyarrow()
    assert out.column("v").to_pylist() == [4, 5, 6]


def test_getitem_without_expressions_explains_itself(lazy):
    if _HAVE_EXPRESSIONS:
        pytest.skip("expression bindings are present")
    with pytest.raises(RuntimeError, match="_expr_column"):
        lazy["v"]
