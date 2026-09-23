"""What the two cost-based rules actually do to a multi-join plan.

`golden/cases/join_three_way_*.mojo` and `join_four_way_chain.mojo` assert that
the **answer** survives `SelectBuildSide` and `JoinReassociation`. They cannot
assert that either rule *ran*: a rule that silently stopped firing would leave
every one of those cases green and the corpus would keep the coverage while
losing the question. That is what this file is for — the same five plans over
the same fixture bytes, checked on their **shape**.

The plans are built through the Python frontend rather than transcribed,
because `LazyTable` and `DynRelation` reach the same nodes and `optimize()` is
`AllRules` in both lanes. A shape assertion reads off `explain()`:

* a left-deep chain renders ``Join(Join(...``; the reassociated form does not,
  since its top join's left input is a source rather than a join, and
* ``Join.write_to`` prints ``build=right`` only when the side is *not* the
  default, so counting the occurrences counts the flips.

The fixtures are the golden corpus's own files, read through
`devkit.golden.corpus` — the same bytes the Mojo lane reads, so a divergence
between this file and a golden case can never be the data.
"""

import pytest

import marrow as ma
from marrow import col

from devkit.golden import corpus


@pytest.fixture(scope="module")
def fixture_tables():
    """The golden fixtures as lazy in-memory tables, keyed by name."""
    fixtures = corpus().fixtures
    fixtures.write()
    return {
        name: ma.memtable(ma.read_ipc_file(str(fixtures.path(name)))[0])
        for name in fixtures.names
    }


def rows(plan):
    return plan.collect(1).to_pylist()


def builds_on_the_right(text):
    return text.count("build=right")


# ---------------------------------------------------------------------------
# The five plans, mirroring the golden cases by name
# ---------------------------------------------------------------------------


def _three_way_chain(t):
    """`golden/cases/join_three_way_chain.mojo` — and `_reassociated`, which is
    the same plan with `.optimize()` on the end."""
    return (
        t["sales"]
        .join(t["emp"], [4], [0], kind="inner")
        .join(t["dept"], [6], [0], kind="inner")
        .project(
            ["ref", "qty", "price", "eid", "dept", "did"],
            [col("ref"), col("qty"), col("price"), col("eid"), col("dept"), col("did")],
        )
        .sort_by([col("ref"), col("qty")], [True, True])
    )


def _key_on_first_input(t):
    """`golden/cases/join_three_way_key_on_first_input.mojo`."""
    return (
        t["emp"]
        .join(t["dept"], [1], [0], kind="inner")
        .join(t["sales"], [0], [4], kind="inner")
        .project(
            ["eid", "dept", "did", "ref", "qty"],
            [col("eid"), col("dept"), col("did"), col("ref"), col("qty")],
        )
        .sort_by([col("eid"), col("qty")], [True, True])
    )


def _inner_then_left(t):
    """`golden/cases/join_three_way_inner_then_left.mojo`."""
    return (
        t["sales"]
        .join(t["emp"], [4], [0], kind="inner")
        .join(t["edges"], [5], [0], kind="left")
        .project(
            ["ref", "qty", "eid", "dept", "i", "j"],
            [col("ref"), col("qty"), col("eid"), col("dept"), col("i"), col("j")],
        )
        .sort_by([col("ref"), col("qty")], [True, True])
    )


def _left_then_inner(t):
    """`golden/cases/join_three_way_left_then_inner.mojo`."""
    mates = t["emp"].rename(["eid", "dept"], ["meid", "mdept"])
    return (
        t["emp"]
        .join(t["dept"], [1], [0], kind="left")
        .join(mates, [2], [1], kind="inner")
        .project(
            ["eid", "dept", "did", "meid"],
            [col("eid"), col("dept"), col("did"), col("meid")],
        )
        .sort_by([col("eid"), col("meid")], [True, True])
    )


def _four_way_chain(t):
    """`golden/cases/join_four_way_chain.mojo`."""
    mates = t["emp"].rename(["eid", "dept"], ["meid", "mdept"])
    return (
        t["sales"]
        .join(t["emp"], [4], [0], kind="inner")
        .join(t["dept"], [6], [0], kind="inner")
        .join(mates, [7], [1], kind="inner")
        .project(
            ["ref", "qty", "eid", "dept", "did", "meid"],
            [col("ref"), col("qty"), col("eid"), col("dept"), col("did"), col("meid")],
        )
        .sort_by([col("ref"), col("qty"), col("meid")], [True, True, True])
    )


# ---------------------------------------------------------------------------
# JoinReassociation -- fires
# ---------------------------------------------------------------------------


def test_three_way_inner_chain_turns_right_deep(fixture_tables):
    """`(sales ⋈ emp) ⋈ dept` -> `sales ⋈ (emp ⋈ dept)`.

    The rule's whole signature: the top join's left input stops being a join.
    """
    plan = _three_way_chain(fixture_tables)
    assert "Join(Join(" in plan.explain()
    assert "Join(Join(" not in plan.optimize().explain()


def test_reassociation_does_not_change_the_answer(fixture_tables):
    """What `golden/cases/join_three_way_reassociated.mojo` asserts against
    DuckDB, asserted here against the unrewritten plan."""
    plan = _three_way_chain(fixture_tables)
    assert rows(plan.optimize()) == rows(plan)


def test_reassociation_leaves_both_joins_indexing_their_right_input(fixture_tables):
    """`SelectBuildSide` runs immediately after and tunes both new joins — so
    three physical decisions separate the plan that runs from the one written,
    not one."""
    plan = _three_way_chain(fixture_tables)
    assert builds_on_the_right(plan.explain()) == 0
    assert builds_on_the_right(plan.optimize().explain()) == 2


def test_four_way_chain_lifts_the_first_input_out(fixture_tables):
    """Three joins and two associations to choose between. The outermost pair
    reassociates, so `Join(Join(Join(` — the left-deep spine — goes."""
    plan = _four_way_chain(fixture_tables)
    text = plan.explain()
    assert "Join(Join(Join(" in text
    optimized = plan.optimize().explain()
    assert "Join(Join(Join(" not in optimized
    assert optimized.count("Join(") == text.count("Join(")
    assert rows(plan.optimize()) == rows(plan)


# ---------------------------------------------------------------------------
# JoinReassociation -- declines
# ---------------------------------------------------------------------------


def test_an_outer_key_naming_the_first_input_declines(fixture_tables):
    """`(emp ⋈ dept) ⋈ sales ON emp.eid = sales.ref`.

    `eid` has nowhere to go in `dept ⋈ sales`, so the plan stays left-deep —
    and it stays left-deep on the *guard*, not on the cost: `SelectBuildSide`
    still tunes it, which is the evidence that the cost was known and the rule
    reached its name check rather than its `Approx.known()` one.
    """
    plan = _key_on_first_input(fixture_tables)
    optimized = plan.optimize().explain()
    assert "Join(Join(" in optimized
    assert builds_on_the_right(optimized) == 1
    assert rows(plan.optimize()) == rows(plan)


def test_a_left_outer_join_on_top_declines(fixture_tables):
    """`(sales ⋈ emp) LEFT JOIN edges` — declined on the outer join's kind,
    while `SelectBuildSide` still flips that same join to index `edges`."""
    plan = _inner_then_left(fixture_tables)
    optimized = plan.optimize().explain()
    assert "Join(Join(" in optimized
    assert builds_on_the_right(optimized) == 1
    assert rows(plan.optimize()) == rows(plan)


def test_a_left_outer_join_underneath_declines(fixture_tables):
    """`(emp LEFT JOIN dept) ⋈ emp` — declined on the *inner* join's kind.

    Reassociating this one would change the answer rather than the cost: the
    two `emp` rows the LEFT join widens are dropped by the inner join here and
    would be widened a second time in `emp LEFT JOIN (dept ⋈ emp)`.
    """
    plan = _left_then_inner(fixture_tables)
    optimized = plan.optimize().explain()
    assert "Join(Join(" in optimized
    assert len(rows(plan.optimize())) == 5
    assert rows(plan.optimize()) == rows(plan)


# ---------------------------------------------------------------------------
# How far the rule can see
# ---------------------------------------------------------------------------


def test_a_many_to_many_intermediate_is_invisible_without_a_distinct_count():
    """The shape `JoinReassociation` exists for, which it cannot currently see.

    `A ⋈ B` here really produces 40x its inputs and `B ⋈ C` produces 25, so the
    right-deep association is the only sane one — `bench_join_multi.py` measures
    it at two orders of magnitude. But `InMemoryTable.estimate` records no
    distinct count, so `ColumnEstimate.max_distinct` falls back to the row
    count, the containment denominator is `max(|L|, |R|)` and the explosion is
    estimated at `min(|L|, |R|)` = 1,000 rows. The rule declines on what it can
    see, and only `SelectBuildSide` fires.

    This pins the *reach*, not the design: wiring a source that records an NDV
    — a Parquet footer, through `ParquetScan.with_statistics` — turns this red,
    and that is the point of asserting it.
    """
    keys, bridge = 25, 1_000
    n = 20_000
    a = ma.memtable(
        ma.record_batch(
            {
                "ak": ma.array([i % keys for i in range(n)], type=ma.int64()),
                "av": ma.array(list(range(n)), type=ma.int64()),
            }
        )
    )
    b = ma.memtable(
        ma.record_batch(
            {
                "bk": ma.array([i % keys for i in range(bridge)], type=ma.int64()),
                "bc": ma.array(list(range(bridge)), type=ma.int64()),
            }
        )
    )
    c = ma.memtable(
        ma.record_batch(
            {
                "cc": ma.array(
                    [i if i < keys else bridge + i for i in range(bridge)],
                    type=ma.int64(),
                ),
                "cv": ma.array(list(range(bridge)), type=ma.int64()),
            }
        )
    )
    plan = a.join(b, left_on="ak", right_on="bk").join(c, left_on="bc", right_on="cc")
    optimized = plan.optimize().explain()
    assert "Join(Join(" in optimized, "reassociation now reaches this — update the test"
    assert builds_on_the_right(optimized) == 2
