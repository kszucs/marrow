"""Tests for the Python expression surface — ``col`` / ``lit`` / ``Column``.

Expressions are checked two ways: that they *render* (the tree was built as
written) and that they *evaluate* (``Column.execute`` against a
``RecordBatch``). The plan layer is not involved.
"""

import pytest

import marrow as ma
from marrow import Aggregate, Column, col, if_else, lit


@pytest.fixture
def batch():
    return ma.record_batch(
        {
            "a": ma.array([1, 2, 3, 4]),
            "b": ma.array([10, 20, 30, 40]),
            "f": ma.array([1.5, -2.5, 4.0, 0.0]),
            "s": ma.array(["ax", "bx", "cy", None]),
        }
    )


@pytest.fixture
def timestamps():
    """A timestamp column, built by casting — ``array()`` cannot infer one."""
    ints = ma.record_batch({"t": ma.array([1615734566, 1656633600, 1704067199, 0])})
    return ints, col("t").cast(ma.timestamp("s", None))


# ── construction and rendering ─────────────────────────────────────────────


def test_col_builds_a_column():
    e = col("a")
    assert isinstance(e, Column)
    assert e.name() == "a"
    assert e.render() == "a"
    assert str(e) == "a"
    assert repr(e) == "<marrow.Column: a>"


def test_lit_renders_as_a_literal():
    assert lit(1).render() == "literal"
    assert lit("x").render() == "literal"
    assert lit(2.5).render() == "literal"


def test_lit_is_idempotent_on_a_column():
    e = col("a")
    assert lit(e) is e


def test_arithmetic_renders_the_tree():
    assert str(col("a") + lit(1)) == "add(a, literal)"
    assert str((col("a") + 1) * 2) == "multiply(add(a, literal), literal)"
    assert str(col("a") - col("b")) == "subtract(a, b)"
    assert str(-col("a")) == "negate(a)"


def test_comparison_renders_the_tree():
    assert str(col("a") > 1) == "greater(a, literal)"
    assert str(col("a") == 1) == "equal(a, literal)"
    assert str(col("a") != 1) == "not_equal(a, literal)"
    assert str((col("a") > 1) & (col("b") < 30)) == (
        "and(greater(a, literal), less(b, literal))"
    )


def test_referenced_columns_is_deduplicated_and_ordered():
    assert col("a").referenced_columns() == ["a"]
    assert (col("a") + col("b") + col("a")).referenced_columns() == ["a", "b"]


def test_non_column_expression_has_no_name():
    assert (col("a") + 1).name() == ""


def test_column_is_not_hashable():
    """`__eq__` builds a predicate, so a Column cannot also be a dict key."""
    with pytest.raises(TypeError):
        hash(col("a"))


# ── evaluation ─────────────────────────────────────────────────────────────


def test_arithmetic_evaluates(batch):
    assert (col("a") + 1).execute(batch).to_pylist() == [2, 3, 4, 5]
    assert (col("a") * col("b")).execute(batch).to_pylist() == [10, 40, 90, 160]
    assert (col("b") - col("a")).execute(batch).to_pylist() == [9, 18, 27, 36]
    assert (col("a") / 2).execute(batch).to_pylist() == [0.5, 1.0, 1.5, 2.0]
    assert (col("b") // 4).execute(batch).to_pylist() == [2, 5, 7, 10]
    assert (col("b") % 7).execute(batch).to_pylist() == [3, 6, 2, 5]
    assert (col("a") ** 2).execute(batch).to_pylist() == [1.0, 4.0, 9.0, 16.0]
    assert (-col("a")).execute(batch).to_pylist() == [-1, -2, -3, -4]


def test_reflected_operators_evaluate(batch):
    """`10 - col("a")` must not silently become `col("a") - 10`."""
    assert (10 - col("a")).execute(batch).to_pylist() == [9, 8, 7, 6]
    assert (1 + col("a")).execute(batch).to_pylist() == [2, 3, 4, 5]
    assert (100 // col("b")).execute(batch).to_pylist() == [10, 5, 3, 2]


def test_comparison_evaluates(batch):
    assert (col("a") > 2).execute(batch).to_pylist() == [
        False,
        False,
        True,
        True,
    ]
    assert (col("a") == 2).execute(batch).to_pylist() == [
        False,
        True,
        False,
        False,
    ]
    assert (col("a") <= 2).execute(batch).to_pylist() == [
        True,
        True,
        False,
        False,
    ]


def test_boolean_combinators_evaluate(batch):
    both = (col("a") > 1) & (col("b") < 40)
    assert both.execute(batch).to_pylist() == [False, True, True, False]
    either = (col("a") > 3) | (col("b") < 20)
    assert either.execute(batch).to_pylist() == [True, False, False, True]
    assert (~(col("a") > 2)).execute(batch).to_pylist() == [
        True,
        True,
        False,
        False,
    ]


def test_math_evaluates(batch):
    assert col("f").abs().execute(batch).to_pylist() == [1.5, 2.5, 4.0, 0.0]
    assert abs(col("f")).execute(batch).to_pylist() == [1.5, 2.5, 4.0, 0.0]
    assert col("f").floor().execute(batch).to_pylist() == [1.0, -3.0, 4.0, 0.0]
    assert col("f").ceil().execute(batch).to_pylist() == [2.0, -2.0, 4.0, 0.0]
    assert col("f").sign().execute(batch).to_pylist() == [1.0, -1.0, 1.0, 0.0]
    assert col("b").sqrt().execute(batch).to_pylist()[0] == pytest.approx(
        3.1622776601683795
    )
    assert col("a").exp().execute(batch).to_pylist()[0] == pytest.approx(2.718281828)
    assert col("b").ln().execute(batch).to_pylist()[0] == pytest.approx(2.302585092)


def test_string_kernels_evaluate(batch):
    assert col("s").upper().execute(batch).to_pylist() == [
        "AX",
        "BX",
        "CY",
        None,
    ]
    assert col("s").length().execute(batch).to_pylist() == [2, 2, 2, None]
    assert col("s").reverse().execute(batch).to_pylist() == [
        "xa",
        "xb",
        "yc",
        None,
    ]
    assert col("s").startswith("a").execute(batch).to_pylist() == [
        True,
        False,
        False,
        None,
    ]
    assert col("s").endswith("y").execute(batch).to_pylist() == [
        False,
        False,
        True,
        None,
    ]
    assert col("s").contains("x").execute(batch).to_pylist() == [
        True,
        True,
        False,
        None,
    ]


def test_like_and_ilike_take_a_raw_pattern(batch):
    assert col("s").like("%x").execute(batch).to_pylist() == [
        True,
        True,
        False,
        None,
    ]
    assert col("s").ilike("A%").execute(batch).to_pylist() == [
        True,
        False,
        False,
        None,
    ]


def test_temporal_fields_evaluate(timestamps):
    ints, ts = timestamps
    assert ts.year().execute(ints).to_pylist() == [2021, 2022, 2023, 1970]
    assert ts.month().execute(ints).to_pylist() == [3, 7, 12, 1]
    assert ts.day().execute(ints).to_pylist() == [14, 1, 31, 1]
    assert ts.hour().execute(ints).to_pylist() == [15, 0, 23, 0]
    assert ts.quarter().execute(ints).to_pylist() == [1, 3, 4, 1]
    assert ts.day_of_year().execute(ints).to_pylist() == [73, 182, 365, 1]


def test_date_trunc_truncates(timestamps):
    ints, ts = timestamps
    truncated = ts.date_trunc("month").cast(ma.int64())
    # 2021-03-01, 2022-07-01, 2023-12-01, 1970-01-01 as epoch seconds.
    assert truncated.execute(ints).to_pylist() == [
        1614556800,
        1656633600,
        1701388800,
        0,
    ]


def test_conditional_and_membership_evaluate(batch):
    assert col("s").coalesce(lit("none")).execute(batch).to_pylist() == [
        "ax",
        "bx",
        "cy",
        "none",
    ]
    assert col("a").nullif(lit(2)).execute(batch).to_pylist() == [
        1,
        None,
        3,
        4,
    ]
    assert col("a").isin([1, 3]).execute(batch).to_pylist() == [
        True,
        False,
        True,
        False,
    ]
    assert col("a").isin(ma.array([2, 4])).execute(batch).to_pylist() == [
        False,
        True,
        False,
        True,
    ]


def test_if_else_evaluates(batch):
    picked = if_else(col("a") > 2, col("a"), lit(0))
    assert picked.execute(batch).to_pylist() == [0, 0, 3, 4]


def test_cast_evaluates(batch):
    assert col("a").cast(ma.float64()).execute(batch).to_pylist() == [
        1.0,
        2.0,
        3.0,
        4.0,
    ]


def test_execute_accepts_the_raw_binding(batch):
    """The plan layer hands raw bindings around; `execute` must accept one."""
    assert (col("a") + 1).execute(batch.unwrap()).to_pylist() == [2, 3, 4, 5]


# ── aggregates ─────────────────────────────────────────────────────────────


def test_aggregates_carry_their_function_and_name():
    for name in ("sum", "mean", "product", "min", "max", "count"):
        agg = getattr(col("a"), name)()
        assert isinstance(agg, Aggregate)
        assert agg.function() == name
        assert agg.name() == name
        assert agg.render() == f"{name}(a)"


def test_alias_renames_the_output_column():
    agg = col("amount").sum().alias("total")
    assert agg.function() == "sum"
    assert agg.name() == "total"
    assert agg.render() == "sum(amount) as total"
    assert repr(agg) == "<marrow.Aggregate: sum(amount) as total>"


def test_alias_keyword_is_equivalent_to_the_method():
    assert col("a").sum(alias="t").name() == col("a").sum().alias("t").name()


def test_aggregate_by_name():
    assert col("a").aggregate("mean").function() == "mean"


def test_aggregate_input_is_a_column():
    agg = (col("a") + col("b")).sum()
    assert isinstance(agg.input(), Column)
    assert agg.input().render() == "add(a, b)"


def test_unknown_aggregate_is_accepted_until_the_plan_resolves_it():
    """`DynAgg` only carries the name; resolution happens at plan build."""
    assert col("a").aggregate("nonesuch").function() == "nonesuch"


# ── the wrap/unwrap contract the plan layer depends on ─────────────────────


def test_column_wrap_unwrap_roundtrip():
    e = col("a")
    binding = e.unwrap()
    assert type(binding).__name__ == "Expr"
    assert Column.wrap(binding).render() == "a"


def test_aggregate_wrap_unwrap_roundtrip():
    agg = col("a").sum().alias("total")
    binding = agg.unwrap()
    assert type(binding).__name__ == "Agg"
    assert Aggregate.wrap(binding).name() == "total"


def test_binding_layer_is_strict_about_operands():
    """The Mojo layer coerces nothing — that is `Column`'s job."""
    with pytest.raises(Exception):
        col("a").unwrap().add(1)
