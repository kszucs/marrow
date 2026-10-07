# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""What the cost-based passes actually do to a multi-join plan.

`golden/cases/join_three_way_*.mojo` and `join_four_way_chain.mojo` assert that
the **answer** survives the join search. They cannot assert that it *ran*: a
search that silently stopped firing would leave every one of those cases green
and the corpus would keep the coverage while losing the question. That is what
this file is for — the same five plans over the same fixture bytes, checked on
their **shape**.

The plans are built through the Python frontend rather than transcribed,
because `LazyTable` and `DynRelation` reach the same nodes and `optimize()` is
`AllRules` in both lanes. A shape assertion reads off `explain()`, where a join
chain renders as ``Join(inputs | links | order tree | outputs)``:

* the links are the joins as written, and the ``order`` section the tree the
  planner chose — shown only when it differs from the written one, left-deep
  in link order with each join hashing the side its link names;
* the tree names participants ``#0``, ``#1``, … in the order they were joined,
  so a left-deep tree over three starts ``((`` and one whose top join's left
  input is a participant does not, and
* a join prints ``build=right`` only when the side is *not* the default, so
  counting the occurrences counts the flips.

The fixtures are the golden corpus's own files, read through
`devkit.golden.corpus` — the same bytes the Mojo lane reads, so a divergence
between this file and a golden case can never be the data.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

import marrow as ma
import marrow.tests.strategies  # noqa: F401 -- registers the settings profiles
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


def tree(text):
    """The tree the planner chose for the one chain in an `explain()`
    rendering, or `None` when it kept the written one."""
    sections = text[text.index("Join(") :].split(" | ")
    for section in sections:
        if section.startswith("order "):
            return section[len("order ") :]
    return None


def left_deep(text, joins):
    """Is the tree `joins` joins deep down its left spine? The written one
    always is."""
    chosen = tree(text)
    return chosen is None or chosen.startswith("(" * joins)


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
# The join search -- reorders
# ---------------------------------------------------------------------------


def test_three_way_inner_chain_hashes_both_dimensions(fixture_tables):
    """`(sales ⋈ emp) ⋈ dept` over analysed tables: `sales ⋈ emp` follows a
    unique key, so the written order is already the cheapest, and the search
    builds both joins on their smaller side."""
    plan = _three_way_chain(fixture_tables)
    optimized = plan.optimize().explain()
    assert left_deep(optimized, 2), optimized
    assert builds_on_the_right(optimized) == 2, optimized
    assert rows(plan.optimize()) == rows(plan)


def test_reassociation_does_not_change_the_answer(fixture_tables):
    """What `golden/cases/join_three_way_reassociated.mojo` asserts against
    DuckDB, asserted here against the unrewritten plan."""
    plan = _three_way_chain(fixture_tables)
    assert rows(plan.optimize()) == rows(plan)


def test_reassociation_leaves_both_joins_indexing_their_right_input(fixture_tables):
    """The search picks each join's build side with the tree, and indexes the
    right input of both — so three physical decisions separate the plan that
    runs from the one written, not one."""
    plan = _three_way_chain(fixture_tables)
    assert builds_on_the_right(plan.explain()) == 0
    assert builds_on_the_right(plan.optimize().explain()) == 2


def test_four_way_chain_lifts_the_first_input_out(fixture_tables):
    """Three joins and five trees to choose between; the left-deep spine is
    not the one chosen."""
    plan = _four_way_chain(fixture_tables)
    assert left_deep(plan.explain(), 3)
    optimized = plan.optimize().explain()
    assert not left_deep(optimized, 3), optimized
    assert tree(optimized).count(" inner ") == 3, optimized
    assert rows(plan.optimize()) == rows(plan)


# ---------------------------------------------------------------------------
# The join search -- keeps the written tree
# ---------------------------------------------------------------------------


def test_a_star_keeps_its_written_tree(fixture_tables):
    """`(emp ⋈ dept) ⋈ sales ON emp.eid = sales.ref` — both joins key on `emp`.

    `dept ⋈ sales` has no key, so the only other tree is `(emp ⋈ sales) ⋈
    dept`, and it is not cheaper: the plan stays as written. Both build sides
    flip to the smaller input, which is the evidence that the cost was known
    and the search compared trees rather than declining.
    """
    plan = _key_on_first_input(fixture_tables)
    optimized = plan.optimize().explain()
    assert left_deep(optimized, 2), optimized
    assert builds_on_the_right(optimized) == 2
    assert rows(plan.optimize()) == rows(plan)


def test_a_left_outer_join_on_top_moves_below_the_inner_join(fixture_tables):
    """`(sales ⋈ emp) LEFT JOIN edges ON emp.eid = edges.i` — the LEFT join
    reads only `emp`, its spine, so it may attach `edges` to `emp` before
    `sales` joins: `sales ⋈ (emp ⟕ edges)`, the same rows."""
    plan = _inner_then_left(fixture_tables)
    optimized = plan.optimize().explain()
    assert tree(optimized).startswith("(#0 inner (#1 left outer #2"), optimized
    assert rows(plan.optimize()) == rows(plan)


def test_a_left_outer_join_underneath_declines(fixture_tables):
    """`(emp LEFT JOIN dept) ⋈ emp` — the LEFT join is a leaf of the inner
    join above it, never taken apart.

    Reassociating across it would change the answer rather than the cost: the
    two `emp` rows the LEFT join widens are dropped by the inner join here and
    would be widened a second time in `emp LEFT JOIN (dept ⋈ emp)`.
    """
    plan = _left_then_inner(fixture_tables)
    optimized = plan.optimize().explain()
    assert left_deep(optimized, 2), optimized
    assert len(rows(plan.optimize())) == 5
    assert rows(plan.optimize()) == rows(plan)


# ---------------------------------------------------------------------------
# How far the search can see
# ---------------------------------------------------------------------------


def _many_to_many():
    """`A ⋈ B` really produces 40x its inputs and `B ⋈ C` produces 25, so
    joining `B ⋈ C` first is the only sane order — `bench_join_multi.py`
    measures it at two orders of magnitude."""
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
    return a, b, c


def test_optimize_sees_a_many_to_many_explosion():
    """`optimize()` analyses its sources: 25 distinct keys on each side of
    `A ⋈ B`, so the search joins `B ⋈ C` first and the answer does not move."""
    a, b, c = _many_to_many()
    blind = a.join(b, left_on="ak", right_on="bk").join(c, left_on="bc", right_on="cc")
    optimized = blind.optimize()
    assert not left_deep(optimized.explain(), 2), optimized.explain()
    assert optimized.column_names == blind.column_names
    key = [col(name) for name in blind.column_names]
    order = [True] * len(key)
    assert rows(optimized.sort_by(key, order)) == rows(blind.sort_by(key, order))


# ---------------------------------------------------------------------------
# ibis's names
# ---------------------------------------------------------------------------


def test_an_inner_key_both_sides_name_is_one_column():
    """`on="k"` joins two `k`s the join makes equal, so the chain emits one;
    a name only one side has keys nothing and is renamed by `rname`."""
    left = ma.memtable(
        ma.record_batch(
            {
                "k": ma.array([1, 2, 3], type=ma.int64()),
                "v": ma.array([10, 20, 30], type=ma.int64()),
            }
        )
    )
    right = ma.memtable(
        ma.record_batch(
            {
                "k": ma.array([2, 3, 4], type=ma.int64()),
                "v": ma.array([200, 300, 400], type=ma.int64()),
            }
        )
    )
    assert left.join(right, on="k").column_names == ["k", "v", "v_right"]
    assert left.join(right, on="k", how="left").column_names == [
        "k",
        "v",
        "k_right",
        "v_right",
    ]
    custom = left.join(right, on="k", lname="l_{name}", rname="r_{name}")
    assert custom.column_names == ["k", "l_v", "r_v"]
    assert rows(custom.sort_by([col("k")], [True])) == [
        {"k": 2, "l_v": 20, "r_v": 200},
        {"k": 3, "l_v": 30, "r_v": 300},
    ]
    with pytest.raises(ma.ArrowInvalid, match="two columns would be named 'v'"):
        left.join(right, on="k", rname="{name}")


def _keyed_pair():
    left = ma.memtable(
        ma.record_batch(
            {
                "k": ma.array([1, 2], type=ma.int64()),
                "v": ma.array([10, 20], type=ma.int64()),
                "v_l": ma.array([5, 6], type=ma.int64()),
            }
        )
    )
    right = ma.memtable(
        ma.record_batch(
            {
                "k": ma.array([1, 1, 2], type=ma.int64()),
                "v": ma.array([100, 101, 200], type=ma.int64()),
            }
        )
    )
    return left, right


def test_strictness_any_keeps_one_match_per_probe_row():
    """The left side is hashed and the right probes it: `k = 1` is twice on
    the hashed side, and `strictness="any"` keeps one match per probe row."""
    left, right = _keyed_pair()
    assert len(rows(right.join(left, on="k"))) == 3
    assert len(rows(right.join(left, on="k", strictness="any"))) == 2


def test_an_unknown_strictness_is_refused():
    left, right = _keyed_pair()
    with pytest.raises(ValueError, match="strictness"):
        left.join(right, on="k", strictness="some")


def test_a_clash_lname_makes_raises_at_the_join():
    """`lname="{name}_l"` renames the left `v` onto the existing `v_l`: two
    output columns would share a name, so the join refuses."""
    left, right = _keyed_pair()
    with pytest.raises(ma.ArrowInvalid, match="two columns would be named 'v_l'"):
        left.join(right, on="k", lname="{name}_l")


# ---------------------------------------------------------------------------
# Every reordering answers as the written chain does
# ---------------------------------------------------------------------------

_KINDS = ["inner", "left", "right", "full", "semi", "anti"]


@st.composite
def join_chains(draw):
    """Three to six small tables joined in a chain, each on its own key to a
    key an earlier join kept, by any kind — inner sometimes `JOIN_ANY` —
    sometimes joining a nested join of the next two tables rather than one,
    with NULL keys, empty tables, a filter over two value columns between
    joins and on top now and then, and statistics now and then. Returns the
    plan and whether its answer is exact: a `JOIN_ANY` join keeps any one
    match."""
    n = draw(st.integers(3, 6))
    tables = []
    for i in range(n):
        rows = draw(st.integers(0, 6))
        key = st.one_of(st.none(), st.integers(0, 3))
        tables.append(
            ma.memtable(
                ma.record_batch(
                    {
                        f"k{i}": ma.array(
                            draw(st.lists(key, min_size=rows, max_size=rows)),
                            type=ma.int64(),
                        ),
                        f"v{i}": ma.array(
                            draw(
                                st.lists(
                                    st.integers(0, 20), min_size=rows, max_size=rows
                                )
                            ),
                            type=ma.int64(),
                        ),
                    }
                )
            )
        )

    def joined(left, right, left_on, right_on):
        kind = draw(st.sampled_from(_KINDS))
        strictness = "any" if kind == "inner" and draw(st.booleans()) else "all"
        plan = left.join(
            right,
            left_on=left_on,
            right_on=right_on,
            how=kind,
            strictness=strictness,
        )
        return plan, kind, strictness == "all"

    def filtered(plan, values):
        if len(values) >= 2 and draw(st.booleans()):
            a, b = draw(st.permutations(values))[:2]
            plan = plan.filter(col(a) < col(b))
        return plan

    plan = tables[0]
    keys, values, exact = ["k0"], ["v0"], True
    i = 1
    while i < n:
        right, own = tables[i], [i]
        if i + 1 < n and draw(st.booleans()):
            right, kind, all_ = joined(tables[i], tables[i + 1], f"k{i}", f"k{i + 1}")
            exact = exact and all_
            if kind not in ("semi", "anti"):
                own.append(i + 1)
            i += 1
        plan, kind, all_ = joined(
            plan, right, draw(st.sampled_from(keys)), f"k{own[0]}"
        )
        exact = exact and all_
        if kind not in ("semi", "anti"):
            keys.extend(f"k{j}" for j in own)
            values.extend(f"v{j}" for j in own)
        plan = filtered(plan, values)
        i += 1
    if draw(st.booleans()):
        plan = plan.analyze()
    return plan, exact


def _bag(batch):
    """The rows as a sorted list of tuples, NULL before any value."""
    return sorted(
        (tuple(row.values()) for row in batch.to_pylist()),
        key=lambda row: tuple((v is not None, v or 0) for v in row),
    )


@given(join_chains())
def test_every_reordering_answers_as_written(chain):
    """The optimized plan — join order, build sides, filters moved into and
    within the chain — returns the written plan's rows, as a bag."""
    plan, exact = chain
    written = _bag(plan.collect(1))
    optimized = _bag(plan.optimize().collect(1))
    if exact:
        assert optimized == written, plan.optimize().explain()
    else:
        assert len(optimized) == len(written), plan.optimize().explain()
