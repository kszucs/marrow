"""The lazy relational frontend — ``LazyTable`` over the bound ``Plan``.

Two things are being checked, and they are different:

- that a verb **builds the plan node it says it does**, readable off
  ``explain()`` without running anything, and
- that ``collect()`` then produces the rows a SQL engine would.

The second is what matters; the first is what tells you *which* verb broke when
it does not. PyArrow appears only as the interop oracle.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import marrow as ma
from marrow import col, count_star, lit


@pytest.fixture
def table():
    """Four rows, one null, two regions."""
    return ma.memtable(
        ma.record_batch(
            {
                "region": ma.array(["east", "west", "east", "west"]),
                "qty": ma.array([1, 2, None, 4], type=ma.int64()),
                "price": ma.array([10, 20, 30, 40], type=ma.int64()),
            }
        )
    )


def rows(lazy_table):
    """Collect and return a list of row dicts."""
    return lazy_table.collect().to_pylist()


# ── construction and introspection ─────────────────────────────────────────


def test_memtable_wraps_a_batch(table):
    assert table.column_names == ["region", "qty", "price"]
    assert "InMemoryTable" in table.explain()


def test_a_plan_is_a_reusable_template(table):
    """Every verb returns a new plan and the old one still runs — a
    ``DynRelation`` is an immutable description, so nothing is consumed."""
    filtered = table.filter(col("qty") > 1)
    assert len(rows(filtered)) == 2
    assert len(rows(filtered)) == 2
    assert len(rows(table)) == 4


def test_getitem_is_a_column_reference(table):
    assert table["qty"].render() == "qty"


def test_explain_renders_the_whole_tree(table):
    """Not just the root: a ``Filter`` names the node it filters."""
    text = table.filter(col("qty") > 1).limit(2).explain()
    assert "Limit" in text
    assert "Filter" in text
    assert "InMemoryTable" in text


def test_str_and_repr_reach_the_plan_not_the_binding_repr(table):
    """``str(binding)`` would return ``"<marrow.Plan: ...>"`` — the type has no
    ``tp_str`` slot, so it falls back to ``tp_repr``. ``LazyTable`` calls
    ``__str__()`` explicitly to get past that."""
    assert not str(table).startswith("<marrow.Plan")
    assert repr(table).startswith("LazyTable\n")


# ── projection verbs ────────────────────────────────────────────────────────


def test_select_keeps_the_named_columns_in_order(table):
    assert table.select("price", "region").column_names == ["price", "region"]


def test_select_accepts_a_list(table):
    assert table.select(["price"]).column_names == ["price"]


def test_drop_keeps_input_order(table):
    assert table.drop("qty").column_names == ["region", "price"]


def test_drop_rejects_an_unknown_column(table):
    """A typo is otherwise silent — the column it meant to remove survives."""
    with pytest.raises(Exception, match="nope"):
        table.drop("nope")


def test_rename_leaves_the_other_columns_in_place(table):
    renamed = table.rename({"qty": "quantity"})
    assert renamed.column_names == ["region", "quantity", "price"]


def test_project_replaces_the_whole_projection(table):
    projected = table.project(total=col("qty") * col("price"))
    assert projected.column_names == ["total"]
    assert rows(projected) == [
        {"total": 10},
        {"total": 40},
        {"total": None},
        {"total": 160},
    ]


def test_project_accepts_two_parallel_lists(table):
    """The spelling both lanes share — Mojo has no ``**kwargs``."""
    a = table.project(total=col("qty") * col("price"))
    b = table.project(["total"], [col("qty") * col("price")])
    assert rows(a) == rows(b)


def test_project_rejects_both_forms_at_once(table):
    with pytest.raises(TypeError):
        table.project(["a"], [col("qty")], b=col("price"))


def test_with_columns_appends_and_replaces_in_place(table):
    out = table.with_columns(total=col("qty") * col("price"))
    assert out.column_names == ["region", "qty", "price", "total"]

    replaced = table.with_columns(qty=col("qty") + 100)
    assert replaced.column_names == ["region", "qty", "price"]
    assert [r["qty"] for r in rows(replaced)] == [101, 102, None, 104]


def test_mutate_is_with_columns(table):
    assert (
        table.mutate(x=lit(1)).column_names == table.with_columns(x=lit(1)).column_names
    )


# ── filter, sort, limit ─────────────────────────────────────────────────────


def test_filter_keeps_matching_rows(table):
    assert [r["qty"] for r in rows(table.filter(col("qty") > 1))] == [2, 4]


def test_filter_drops_null_predicate_rows(table):
    """A null predicate is not true, so the row goes — SQL's rule."""
    assert len(rows(table.filter(col("qty").is_null()))) == 1


def test_filter_composes_with_boolean_operators(table):
    out = table.filter((col("qty") > 1) & (col("region") == lit("west")))
    assert [r["price"] for r in rows(out)] == [20, 40]


def test_order_by_ascending_and_descending(table):
    assert [r["price"] for r in rows(table.order_by(("price", "descending")))] == [
        40,
        30,
        20,
        10,
    ]
    assert [r["price"] for r in rows(table.order_by("price"))] == [10, 20, 30, 40]


def test_order_by_accepts_a_bool_direction(table):
    assert rows(table.order_by(("price", False))) == rows(
        table.order_by(("price", "descending"))
    )


def test_sort_by_is_order_by(table):
    assert rows(table.sort_by("price")) == rows(table.order_by("price"))


def test_order_by_needs_a_key(table):
    with pytest.raises(ValueError):
        table.order_by()


def test_limit_and_head(table):
    assert len(rows(table.limit(2))) == 2
    assert len(rows(table.head(3))) == 3
    assert [r["price"] for r in rows(table.limit(2, offset=1))] == [20, 30]


# ── aggregation ─────────────────────────────────────────────────────────────


def test_grouped_aggregate_with_keyword_specs(table):
    out = rows(table.aggregate(by=["region"], total=("sum", "price")))
    assert sorted((r["region"], r["total"]) for r in out) == [
        ("east", 40),
        ("west", 60),
    ]


def test_grouped_aggregate_with_expression_aggregates(table):
    out = rows(table.aggregate(by=["region"], total=col("price").sum()))
    assert sorted((r["region"], r["total"]) for r in out) == [
        ("east", 40),
        ("west", 60),
    ]


def test_positional_aggregates_carry_their_own_alias(table):
    out = rows(table.aggregate(["region"], col("price").sum().alias("t")))
    assert sorted(r["t"] for r in out) == [40, 60]


def test_ungrouped_aggregate_is_one_implicit_group(table):
    out = rows(table.aggregate(total=("sum", "price")))
    assert out == [{"total": 100}]


def test_count_star_counts_rows_and_count_counts_values(table):
    """The two disagree on ``qty``, which carries a null."""
    out = rows(table.aggregate(n=count_star(), c=col("qty").count()))
    assert out == [{"n": 4, "c": 3}]


def test_keys_with_no_aggregates_is_select_distinct(table):
    out = rows(table.aggregate(by=["region"]))
    assert sorted(r["region"] for r in out) == ["east", "west"]


def test_aggregate_needs_a_key_or_an_aggregate(table):
    with pytest.raises(ValueError):
        table.aggregate()


def test_a_bare_column_is_not_an_aggregate(table):
    """``total=col("price")`` is a common slip and used to reach Mojo as an
    unaliasable value; it is caught here with the name it was given."""
    with pytest.raises(ValueError, match="not an aggregate"):
        table.aggregate(by=["region"], total=col("price"))


def test_a_bare_function_name_is_ambiguous(table):
    with pytest.raises(ValueError, match="ambiguous"):
        table.aggregate(by=["region"], n="count")


def test_aggregate_spec_arity(table):
    with pytest.raises(ValueError, match=r"\(func, column\)"):
        table.aggregate(by=["region"], total=("sum", "price", "extra"))


# ── join ────────────────────────────────────────────────────────────────────


@pytest.fixture
def regions():
    return ma.memtable(
        ma.record_batch(
            {
                "region": ma.array(["east", "west"]),
                "manager": ma.array(["ann", "bo"]),
            }
        )
    )


def test_inner_join_on_a_shared_name(table, regions):
    out = rows(table.join(regions, on="region"))
    assert len(out) == 4
    assert {r["manager"] for r in out} == {"ann", "bo"}


def test_join_with_separate_key_names(table, regions):
    out = rows(table.join(regions, left_on="region", right_on="region"))
    assert len(out) == 4


def test_join_needs_keys(table, regions):
    with pytest.raises(ValueError, match="pass `on`"):
        table.join(regions)


def test_join_rejects_an_unknown_key(table, regions):
    """Names are resolved to indices here, where the schema is in hand, so the
    message can say which side it looked in."""
    with pytest.raises(ValueError, match="left key 'nope'"):
        table.join(regions, on="nope")


def test_join_rejects_mismatched_key_counts(table, regions):
    with pytest.raises(ValueError, match="left keys"):
        table.join(regions, left_on=["region", "qty"], right_on=["region"])


# ── execution ───────────────────────────────────────────────────────────────


def test_collect_returns_an_eager_record_batch(table):
    out = table.collect()
    assert isinstance(out, ma.RecordBatch)
    # `num_rows` is a method here, not a PyArrow-style property.
    assert out.num_rows == 4


@pytest.mark.parametrize("num_threads", [0, 1, 4])
def test_collect_honours_the_worker_budget(table, num_threads):
    """0 auto, 1 serial, N forced — the eager surface's sentinel set. The
    answer must not depend on it."""
    out = rows(table.aggregate(by=["region"], total=("sum", "price")))
    got = table.aggregate(by=["region"], total=("sum", "price")).collect(num_threads)
    assert sorted(r["total"] for r in got.to_pylist()) == sorted(
        r["total"] for r in out
    )


def test_to_pyarrow_round_trips_through_the_c_data_interface(table):
    out = table.to_pyarrow()
    assert isinstance(out, pa.RecordBatch)
    assert out.column_names == ["region", "qty", "price"]
    assert out.column("price").to_pylist() == [10, 20, 30, 40]


# ── parquet ─────────────────────────────────────────────────────────────────


@pytest.fixture
def parquet_file(tmp_path):
    path = tmp_path / "sales.parquet"
    pq.write_table(
        pa.table(
            {
                "region": ["east", "west", "east", "west"],
                "price": [10, 20, 30, 40],
            }
        ),
        path,
    )
    return path


def test_read_parquet_infers_the_schema_from_the_footer(parquet_file):
    """Metadata only, no column data — which is why no schema is required."""
    t = ma.read_parquet(parquet_file)
    assert t.column_names == ["region", "price"]
    assert "ParquetScan" in t.explain()


def test_read_parquet_executes_a_full_query(parquet_file):
    t = ma.read_parquet(parquet_file)
    out = rows(
        t.filter(col("price") > 15)
        .aggregate(by=["region"], total=("sum", "price"))
        .order_by("region")
    )
    assert out == [
        {"region": "east", "total": 30},
        {"region": "west", "total": 60},
    ]


def test_read_parquet_accepts_a_path_string(parquet_file):
    assert ma.read_parquet(str(parquet_file)).column_names == [
        "region",
        "price",
    ]


# ---------------------------------------------------------------------------
# optimize() and batches()
# ---------------------------------------------------------------------------


def test_optimize_eliminates_a_constant_true_filter(table):
    plan = table.filter(ma.lit(True))
    assert "Filter" in plan.explain()
    assert "Filter" not in plan.optimize().explain()


def test_optimize_splits_a_conjunction(table):
    """Needs both halves of the predicate fix: ``Plan.filter`` keeping the
    concrete ``RuntimeValue``, and ``RuntimeValue.conjuncts`` splitting on
    ``and``. With either missing this comes back as one compound Filter."""
    plan = table.filter((col("qty") > 1) & (col("price") < 50))
    assert plan.optimize().explain().count("Filter") == 2


def test_optimize_splits_a_three_way_conjunction(table):
    plan = table.filter((col("qty") > 1) & (col("price") < 50) & (col("qty") < 4))
    assert plan.optimize().explain().count("Filter") == 3


def test_optimize_does_not_change_the_answer(table):
    plan = table.filter((col("qty") > 1) & (col("price") < 50))
    assert rows(plan.optimize()) == rows(plan)


def test_optimize_removes_a_no_op_projection(table):
    plan = table.select("region", "qty", "price").filter(col("qty") > 2)
    assert "Project" in plan.explain()
    assert "Project" not in plan.optimize().explain()


def test_optimize_returns_a_composable_plan(table):
    """The result is an ordinary plan, not a terminal — you can keep going."""
    plan = table.filter(col("qty") > 1).optimize().select("region")
    # qty is [1, 2, None, 4]; only rows 1 and 3 pass, and both are "west".
    assert rows(plan) == [{"region": "west"}, {"region": "west"}]


def test_batches_returns_the_engines_own_batches(table):
    assert sum(b.num_rows for b in table.batches()) == table.collect().num_rows


def test_batches_agrees_with_collect_under_a_filter(table):
    plan = table.filter(col("qty") > 1)
    assert sum(b.num_rows for b in plan.batches()) == plan.collect().num_rows


def test_to_table_keeps_one_chunk_per_batch(table):
    out = table.to_table()
    assert out.num_rows == table.collect().num_rows
    assert out.column(0).num_chunks == len(table.batches())


# ---------------------------------------------------------------------------
# Window functions
# ---------------------------------------------------------------------------


@pytest.fixture
def windowed():
    """Two partitions of two rows, no ties, so every window verb is legible."""
    return ma.memtable(
        ma.record_batch(
            {
                "k": ma.array(["a", "a", "b", "b"]),
                "v": ma.array([1, 2, 3, 4], type=ma.int64()),
            }
        )
    )


def test_row_number_numbers_the_ordering(windowed):
    out = windowed.with_columns(rn=ma.row_number().over(order_by=[col("v")]))
    assert [r["rn"] for r in rows(out)] == [1, 2, 3, 4]


def test_rank_skips_a_gap_and_dense_rank_does_not(windowed):
    out = windowed.with_columns(
        r=ma.rank().over(order_by=[col("k")]),
        d=ma.dense_rank().over(order_by=[col("k")]),
    )
    assert [x["r"] for x in rows(out)] == [1, 1, 3, 3]
    assert [x["d"] for x in rows(out)] == [1, 1, 2, 2]


def test_lag_and_lead_read_neighbouring_rows(windowed):
    out = windowed.with_columns(
        prev=col("v").lag().over(order_by=[col("v")]),
        nxt=col("v").lead().over(order_by=[col("v")]),
    )
    assert [x["prev"] for x in rows(out)] == [None, 1, 2, 3]
    assert [x["nxt"] for x in rows(out)] == [2, 3, 4, None]


def test_first_value_is_constant_and_last_value_is_the_current_row(windowed):
    """The frame gotcha: under the default frame `last_value` is the *current*
    row, because the frame ends there."""
    out = windowed.with_columns(
        f=col("v").first_value().over(order_by=[col("v")]),
        l=col("v").last_value().over(order_by=[col("v")]),
    )
    assert [x["f"] for x in rows(out)] == [1, 1, 1, 1]
    assert [x["l"] for x in rows(out)] == [1, 2, 3, 4]


def test_an_aggregate_windows_per_partition(windowed):
    out = windowed.with_columns(
        running=col("v").sum().over(partition_by=[col("k")], order_by=[col("v")])
    )
    assert [x["running"] for x in rows(out)] == [1, 3, 3, 7]


def test_an_explicit_rows_frame_slides(windowed):
    """`ROWS` counts rows where the default `RANGE` counts peers."""
    out = windowed.with_columns(
        s=col("v").sum().over(order_by=[col("v")], rows=(-1, 0))
    )
    assert [x["s"] for x in rows(out)] == [1, 3, 5, 7]


def test_a_window_builds_a_window_node(windowed):
    plan = windowed.with_columns(rn=ma.row_number().over(order_by=[col("v")]))
    assert "Window(" in plan.explain()


def test_windowing_a_per_row_value_says_what_to_do_instead(windowed):
    with pytest.raises(TypeError, match="aggregate first"):
        col("v").over(order_by=[col("v")])


def test_windows_and_ordinary_expressions_cannot_share_a_call(windowed):
    """They are two different plan nodes, and which ran first would change the
    answer."""
    with pytest.raises(TypeError, match="not both in one call"):
        windowed.with_columns(
            a=col("v") + 1, rn=ma.row_number().over(order_by=[col("v")])
        )


def test_window_renders_readably(windowed):
    w = ma.row_number().over(partition_by=[col("k")], order_by=[col("v")])
    assert "row_number()" in w.render()
    assert "partition k" in w.render()
    assert w.referenced_columns() == ["k", "v"]
