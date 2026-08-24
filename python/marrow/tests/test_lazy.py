"""Tests for the lazy relational frontend (`marrow.LazyTable`)."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import marrow
from marrow.exprold import _HAVE_EXPRESSIONS

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
    return marrow.memtable(batch)


# ── plan construction ──────────────────────────────────────────────────────


def test_scan_is_lazy(lazy):
    assert isinstance(lazy, marrow.LazyTable)
    assert lazy.column_names == ["k", "v"]
    assert "InMemoryTable" in str(lazy)


def test_repr_shows_the_plan(lazy):
    assert "LazyTable" in repr(lazy)
    assert "InMemoryTable" in repr(lazy)


def test_plan_text_is_the_plan_not_its_repr(lazy):
    """`str(binding)` finds the derived `repr` — `def_method` fills `tp_dict`,
    not `tp_str` — so `explain()` used to return `<marrow.Plan: ...>`."""
    assert not lazy.explain().startswith("<marrow.Plan:")
    assert lazy.explain().startswith("InMemoryTable(")
    assert str(lazy) == lazy.explain()


def test_select(lazy):
    assert lazy.select("k").column_names == ["k"]
    assert lazy.select("v", "k").column_names == ["v", "k"]
    assert lazy.select(["k", "v"]).column_names == ["k", "v"]


def test_drop(lazy):
    assert lazy.drop("k").column_names == ["v"]
    assert lazy.drop(["k", "v"]).column_names == []
    assert lazy.drop("k").to_pyarrow().column("v").to_pylist() == [1, 2, 3, 4, 5, 6]


def test_drop_unknown_column_raises(lazy):
    """The plan node owns the check now, so the message comes from Mojo — the
    same shape every other unknown-column error on this surface has."""
    with pytest.raises(Exception, match="drop: column 'nope' not found"):
        lazy.drop("nope")


def test_rename(lazy):
    renamed = lazy.rename({"v": "value"})
    assert renamed.column_names == ["k", "value"]
    assert renamed.to_pyarrow().column("value").to_pylist() == [1, 2, 3, 4, 5, 6]


def test_rename_leaves_order_alone(lazy):
    """Renaming the *first* column must not move it to the end — the plan node
    rewrites fields in place rather than re-projecting."""
    assert lazy.rename({"k": "key"}).column_names == ["key", "v"]


def test_rename_unknown_or_colliding_raises(lazy):
    with pytest.raises(Exception, match="not found"):
        lazy.rename({"nope": "x"})
    with pytest.raises(Exception, match="duplicate"):
        lazy.rename({"v": "k"})


def test_select_unknown_column_raises(lazy):
    with pytest.raises(Exception):
        lazy.select("nope").collect()


@needs_expressions
def test_select_preserves_the_source_field(tmp_path):
    """A pass-through column keeps its whole `Field`, by whichever verb.

    `select` always copied the input `Field`; `project` used to rebuild a bare
    one from the probed dtype alone, so the same column came out with a
    different schema depending on which verb asked for it — a non-nullable
    input widened to nullable. `project` now copies the source `Field` when the
    expression is a bare column reference, so the two agree.

    Parquet is the reachable source of a non-nullable field from Python:
    `pa.field(..., nullable=False)` survives the round trip.
    """
    path = tmp_path / "nn.parquet"
    schema = pa.schema(
        [pa.field("a", pa.int64(), nullable=False), pa.field("b", pa.int64())]
    )
    pq.write_table(pa.table({"a": [1, 2], "b": [3, 4]}, schema=schema), path)

    t = marrow.read_parquet(path)
    assert pa.schema(t.select("a").schema).field("a").nullable is False
    assert pa.schema(t.project(a=t["a"]).schema).field("a").nullable is False


def test_project_still_probes_a_computed_column(tmp_path):
    """Only a *pass-through* inherits its source field.

    A computed column has no source `Field` to inherit, so its dtype is probed
    and it is nullable — the arithmetic's answer, not the input's.
    """
    path = tmp_path / "nn2.parquet"
    schema = pa.schema([pa.field("a", pa.int64(), nullable=False)])
    pq.write_table(pa.table({"a": [1, 2]}, schema=schema), path)

    t = marrow.read_parquet(path)
    computed = pa.schema(t.project(a=t["a"] + 1).schema).field("a")
    assert computed.nullable is True
    assert computed.type == pa.int64()


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


def test_aggregate_with_keys_only_is_distinct(lazy):
    """Keys and no aggregates is ``SELECT DISTINCT``.

    This used to raise: the binding required an aggregate even though the
    plan layer has always executed the keys-only form, so the one shape
    ``SELECT DISTINCT k`` needs was reachable from Mojo and not from Python.
    """
    out = lazy.aggregate(by=["k"]).order_by("k").to_pyarrow()
    assert out.column_names == ["k"]
    assert out.column("k").to_pylist() == ["a", "b", "c"]


def test_aggregate_needs_a_key_or_an_aggregate(lazy):
    with pytest.raises(ValueError):
        lazy.aggregate()


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
    out = (
        marrow.memtable(batch)
        .join(marrow.memtable(right), on="k")
        .order_by("v")
        .to_pyarrow()
    )
    assert out.column("label").to_pylist() == [
        "Alpha",
        "Beta",
        "Alpha",
        "Beta",
        "Alpha",
    ]


def test_join_needs_keys(batch):
    with pytest.raises(ValueError):
        marrow.memtable(batch).join(marrow.memtable(batch))


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


@needs_expressions
def test_with_columns_appends(lazy):
    out = lazy.with_columns(double=lazy["v"] * 2)
    assert out.column_names == ["k", "v", "double"]
    assert out.to_pyarrow().column("double").to_pylist() == [2, 4, 6, 8, 10, 12]


@needs_expressions
def test_with_columns_replaces_in_place(lazy):
    """A name already in the schema overwrites *at its position* — polars' and
    ibis' rule, and the reason this is not just `project`."""
    out = lazy.with_columns(v=lazy["v"] + 10)
    assert out.column_names == ["k", "v"]
    assert out.to_pyarrow().column("v").to_pylist() == [11, 12, 13, 14, 15, 16]


@needs_expressions
def test_with_columns_expressions_all_see_the_input(lazy):
    """`c` reads the *original* `v`, not the `v` computed one slot earlier."""
    out = lazy.with_columns(v=lazy["v"] + 100, c=lazy["v"] * 2).to_pyarrow()
    assert out.column("v").to_pylist() == [101, 102, 103, 104, 105, 106]
    assert out.column("c").to_pylist() == [2, 4, 6, 8, 10, 12]


@needs_expressions
def test_mutate_is_with_columns(lazy):
    assert lazy.mutate(d=lazy["v"] * 2).column_names == ["k", "v", "d"]


@needs_expressions
def test_with_columns_survives_a_later_aggregate(lazy):
    """The added column is a real plan column, not a rendering: group by it."""
    out = (
        lazy.with_columns(big=lazy["v"] > 3)
        .aggregate(by=["big"], total=("sum", "v"))
        .order_by("big")
        .to_pyarrow()
    )
    assert out.column("big").to_pylist() == [False, True]
    assert out.column("total").to_pylist() == [6, 15]


def test_count_star_counts_rows_not_values():
    """`count_star()` and `("count", col)` disagree on a nullable column, which
    is the entire reason `COUNT(*)` is bound separately."""
    batch = marrow.record_batch(
        {
            "k": marrow.array(["a", "a", "b"]).unwrap(),
            "v": marrow.array([1, None, None]).unwrap(),
        }
    )
    out = (
        marrow.memtable(batch)
        .aggregate(by=["k"], rows=marrow.count_star(), values=("count", "v"))
        .order_by("k")
        .to_pyarrow()
    )
    assert out.column_names == ["k", "rows", "values"]
    assert out.column("rows").to_pylist() == [2, 1]
    assert out.column("values").to_pylist() == [1, 0]


def test_count_star_with_no_group_keys(lazy):
    out = lazy.aggregate(by=[], n=marrow.count_star()).to_pyarrow()
    assert out.column("n").to_pylist() == [6]


def test_count_star_positionally_keeps_its_own_name(lazy):
    """A positional aggregate carries its own alias; `count_star` supplies one,
    so it needs no keyword."""
    out = lazy.aggregate([], marrow.count_star()).to_pyarrow()
    assert out.column_names == ["count_star"]
    assert out.column("count_star").to_pylist() == [6]


def test_getitem_without_expressions_explains_itself(lazy):
    if _HAVE_EXPRESSIONS:
        pytest.skip("expression bindings are present")
    with pytest.raises(RuntimeError, match="_expr_column"):
        lazy["v"]


def test_execution_errors_propagate_instead_of_truncating(lazy):
    """A kernel raising mid-drain must surface, not read as end-of-stream.

    `Processor.pull()` signals exhaustion by raising `Exhausted`, and the drain
    loops caught it with `except Exhausted:`. Mojo's `except` does **not** match
    on type, so that caught *every* error: a predicate that raised was
    indistinguishable from "no more morsels", and `collect()` returned the
    batches accumulated so far and reported success. Here the predicate raises
    `is_in: dtype mismatch` on every morsel, so the old behaviour was a
    confident, empty, wrong answer.
    """
    import marrow as ma

    # int64 value set against an int16 column. `is_in` is decided on the 64-bit
    # hash, so mismatched widths can never match -- the kernel is right to raise.
    narrow = ma.array(pa.array([1, 2], type=pa.int8()))
    predicate = lazy["v"].isin(narrow)
    with pytest.raises(Exception, match="dtype mismatch"):
        lazy.filter(predicate).collect()


# ── thread count must not change the answer ────────────────────────────────
#
# `collect(num_threads=)` reaches `GroupBy`'s thread-local / radix strategies
# and `HashJoin`'s parallel build + probe. A race in any of them produces a
# *wrong answer*, not an error, so these compare the whole result against the
# serial one rather than a row count. Sizes are past `_PARALLEL_MIN_ROWS`
# (60_000) and `_PARALLEL_ALWAYS_ROWS` (200_000) — below them every strategy
# falls back to serial and the comparison proves nothing.

_PARALLEL_ROWS = 250_000


def _sorted_rows(batch, columns):
    d = batch.to_pydict()
    return sorted(zip(*(d[c] for c in columns)))


@pytest.mark.parametrize("num_threads", [2, 4, 8])
def test_group_by_answer_is_independent_of_thread_count(num_threads):
    """Low *and* high cardinality: they pick different grouping strategies."""
    n = _PARALLEL_ROWS
    for groups in (100, n // 2):
        src = marrow.record_batch(
            pa.record_batch(
                {
                    "k": pa.array([i % groups for i in range(n)], type=pa.int32()),
                    "v": pa.array(range(n), type=pa.float64()),
                }
            )
        )
        plan = marrow.memtable(src).aggregate(
            by=["k"], total=("sum", "v"), n=("count", "v")
        )
        serial = pa.record_batch(plan.collect(num_threads=1))
        parallel = pa.record_batch(plan.collect(num_threads=num_threads))
        cols = ["k", "total", "n"]
        assert serial.num_rows == groups
        assert _sorted_rows(serial, cols) == _sorted_rows(parallel, cols)


@pytest.mark.parametrize("num_threads", [2, 4, 8])
def test_join_answer_is_independent_of_thread_count(num_threads):
    n = _PARALLEL_ROWS
    left = marrow.memtable(
        marrow.record_batch(
            pa.record_batch(
                {
                    "k": pa.array(range(n), type=pa.int64()),
                    "v": pa.array(range(n), type=pa.int64()),
                }
            )
        )
    )
    right = marrow.memtable(
        marrow.record_batch(
            pa.record_batch(
                {
                    "k": pa.array([(i * 7) % n for i in range(n)], type=pa.int64()),
                    "w": pa.array(range(n), type=pa.int64()),
                }
            )
        )
    )
    plan = left.join(right, on="k", how="inner")
    serial = pa.record_batch(plan.collect(num_threads=1))
    parallel = pa.record_batch(plan.collect(num_threads=num_threads))
    cols = ["k", "v", "w"]
    assert serial.num_rows == n
    assert _sorted_rows(serial, cols) == _sorted_rows(parallel, cols)


@needs_expressions
def test_filter_answer_is_independent_of_thread_count():
    n = _PARALLEL_ROWS
    src = marrow.record_batch(
        pa.record_batch({"v": pa.array(range(n), type=pa.int64())})
    )
    plan = marrow.memtable(src).filter(marrow.col("v") < marrow.lit(n // 3))
    serial = pa.record_batch(plan.collect(num_threads=1)).to_pydict()
    parallel = pa.record_batch(plan.collect(num_threads=8)).to_pydict()
    assert len(serial["v"]) == n // 3
    assert serial == parallel


@needs_expressions
def test_join_on_differently_named_keys(tmp_path):
    """`left_on`/`right_on` with different names must work.

    The hash join filters collisions with `EqKernel.apply` over the two key
    structs, and a struct's dtype includes its field names — so selecting the
    key columns kept `dept` on one side and `did` on the other, and the kernel
    rejected the pair with `equal: dtype mismatch: struct<dept: int64> vs
    struct<did: int64>`. Only joins whose keys happened to share a name
    worked, which is the rarer case: a foreign key seldom shares its
    referent's name.
    """
    emp = marrow.record_batch(
        {
            "eid": marrow.array([1, 2, 3]).unwrap(),
            "dept": marrow.array([10, 20, 99]).unwrap(),
        }
    )
    dept = marrow.record_batch(
        {
            "did": marrow.array([10, 20]).unwrap(),
            "dname": marrow.array(["eng", "sales"]).unwrap(),
        }
    )
    out = (
        marrow.memtable(emp)
        .join(marrow.memtable(dept), left_on="dept", right_on="did")
        .order_by("eid")
        .collect()
    )
    assert out.num_rows() == 2
    assert out.column(0).to_pylist() == [1, 2]
    assert out.column(3).to_pylist() == ["eng", "sales"]


@needs_expressions
def test_empty_result_is_a_well_formed_batch(batch):
    """A plan that matches nothing still returns one column per schema field.

    `collect()` used to return the schema with an *empty* column list when no
    morsel survived, so `num_columns()` was 0 while the schema named its
    fields. Anything walking the columns by schema index ran off the end, and
    exporting the batch through the C Data interface returned NULL without
    setting an exception — `to_pyarrow()` raised a bare `SystemError`.
    """
    lazy = marrow.memtable(batch)
    empty = lazy.filter(lazy["v"] > 10_000)

    out = empty.collect()
    assert out.num_rows() == 0
    assert out.num_columns() == batch.num_columns()

    exported = empty.to_pyarrow()
    assert exported.num_rows == 0
    assert exported.schema.names == ["k", "v"]
