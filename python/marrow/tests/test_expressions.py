"""The Python expression surface — ``col``, ``lit`` and everything on them.

Every case evaluates the expression against a batch rather than checking how it
renders: a verb wired to the wrong kernel still renders correctly, and the
whole point of this layer is that ``col("a") * 2`` in Python and ``col("a",
int64) * lit(2, int64)`` in Mojo compute the same thing.

PyArrow is the oracle where the semantics are Arrow's.
"""

import pyarrow as pa
import pytest

import marrow as ma
from marrow import col, count_star, if_else, lit


@pytest.fixture
def batch():
    """``a`` carries a null; ``b`` does not; ``s`` is a string column."""
    return ma.record_batch(
        {
            "a": ma.array([1, 2, None, 4], type=ma.int64()),
            "b": ma.array([10, 20, 30, 40], type=ma.int64()),
            "f": ma.array([1.5, -2.5, 3.0, -4.5], type=ma.float64()),
            "s": ma.array(["apple", "Pear ", None, "quince"]),
        }
    )


def run(expr, batch):
    """Evaluate `expr` over `batch` and return a Python list."""
    return expr.execute(batch).to_pylist()


# ── construction and representation ────────────────────────────────────────


def test_col_renders_as_its_name():
    assert col("a").render() == "a"
    assert col("a").name() == "a"
    assert str(col("a")) == "a"
    assert repr(col("a")) == "<marrow.Column: a>"


def test_col_accepts_and_ignores_a_dtype():
    """``col("a", int64)`` is what the Mojo comptime lane *requires*, so the
    runtime lane accepts the same spelling and resolves against the schema."""
    assert col("a", ma.int64()).render() == col("a").render()


def test_referenced_columns_are_deduped_in_first_seen_order():
    """What projection pushdown will read once it exists."""
    expr = (col("b") + col("a")) * col("b")
    assert expr.referenced_columns() == ["b", "a"]


def test_a_non_column_expression_has_no_name():
    assert (col("a") + 1).name() == ""


def test_column_is_unhashable():
    """``==`` builds a predicate, so a Column cannot also be a dict key — the
    same trade-off as ``pyarrow.compute.Expression``."""
    with pytest.raises(TypeError):
        {col("a"): 1}


# ── arithmetic ──────────────────────────────────────────────────────────────


def test_arithmetic_over_two_columns(batch):
    assert run(col("a") + col("b"), batch) == [11, 22, None, 44]
    assert run(col("b") - col("a"), batch) == [9, 18, None, 36]
    assert run(col("a") * col("b"), batch) == [10, 40, None, 160]


def test_arithmetic_coerces_a_python_scalar(batch):
    """``col("a") + 1`` and ``col("a") + lit(1)`` build the same tree."""
    assert run(col("b") + 1, batch) == run(col("b") + lit(1), batch)
    assert run(col("b") + 1, batch) == [11, 21, 31, 41]


def test_reflected_operators(batch):
    assert run(100 - col("b"), batch) == [90, 80, 70, 60]
    assert run(2 * col("b"), batch) == [20, 40, 60, 80]


def test_true_division_is_always_float(batch):
    """``5 / 2`` is 2.5, not 2 — Python's rule, not ``pc.divide``'s."""
    assert run(col("b") / 4, batch) == [2.5, 5.0, 7.5, 10.0]


def test_floor_division_and_modulo(batch):
    assert run(col("b") // 3, batch) == [3, 6, 10, 13]
    assert run(col("b") % 3, batch) == [1, 2, 0, 1]


def test_negation(batch):
    assert run(-col("b"), batch) == [-10, -20, -30, -40]


def test_arithmetic_widens_a_mixed_pair():
    """An int32 column against an int64 literal must widen, not narrow."""
    small = ma.record_batch({"c": ma.array([1, 2], type=ma.int32())})
    assert run(col("c") + lit(2**40), small) == [2**40 + 1, 2**40 + 2]


def test_string_plus_is_concatenation(batch):
    """Over erased operands ``+`` cannot know at build time whether it means
    addition or concatenation, so the runtime dtype decides."""
    assert run(col("s") + lit("!"), batch) == [
        "apple!",
        "Pear !",
        None,
        "quince!",
    ]


# ── comparison and boolean logic ────────────────────────────────────────────


def test_all_six_comparisons(batch):
    assert run(col("a") == 2, batch) == [False, True, None, False]
    assert run(col("a") != 2, batch) == [True, False, None, True]
    assert run(col("a") < 2, batch) == [True, False, None, False]
    assert run(col("a") <= 2, batch) == [True, True, None, False]
    assert run(col("a") > 2, batch) == [False, False, None, True]
    assert run(col("a") >= 2, batch) == [False, True, None, True]


def test_boolean_logic_is_three_valued(batch):
    """Kleene: ``null AND false`` is false, not null."""
    always = col("b") > 0
    maybe = col("a") > 2
    assert run(always & maybe, batch) == [False, False, None, True]
    assert run(always | maybe, batch) == [True, True, True, True]
    assert run(~maybe, batch) == [True, True, None, False]
    assert run(always ^ maybe, batch) == [True, True, None, False]


def test_string_comparison(batch):
    assert run(col("s") < lit("b"), batch) == [True, True, None, False]


# ── math ────────────────────────────────────────────────────────────────────


def test_unary_math(batch):
    assert run(abs(col("f")), batch) == [1.5, 2.5, 3.0, 4.5]
    assert run(col("f").floor(), batch) == [1.0, -3.0, 3.0, -5.0]
    assert run(col("f").ceil(), batch) == [2.0, -2.0, 3.0, -4.0]
    assert run(col("f").trunc(), batch) == [1.0, -2.0, 3.0, -4.0]
    assert run(col("f").sign(), batch) == [1.0, -1.0, 1.0, -1.0]


def test_sqrt_promotes_an_integer_column(batch):
    """``sqrt``/``exp``/``ln`` are float64 out whatever went in."""
    assert run(col("b").sqrt(), batch) == pytest.approx(
        [10**0.5, 20**0.5, 30**0.5, 40**0.5]
    )


def test_round_rejects_ndigits(batch):
    with pytest.raises(NotImplementedError):
        round(col("f"), 2)


# ── strings ─────────────────────────────────────────────────────────────────


def test_case_and_strip(batch):
    assert run(col("s").upper(), batch) == [
        "APPLE",
        "PEAR ",
        None,
        "QUINCE",
    ]
    assert run(col("s").strip(), batch) == ["apple", "Pear", None, "quince"]


def test_length_is_byte_length_and_null_preserving(batch):
    assert run(col("s").length(), batch) == [5, 5, None, 6]


def test_predicates_and_patterns(batch):
    assert run(col("s").startswith(lit("ap")), batch) == [
        True,
        False,
        None,
        False,
    ]
    assert run(col("s").contains(lit("ea")), batch) == [
        False,
        True,
        None,
        False,
    ]
    assert run(col("s").like("%ce"), batch) == [False, False, None, True]
    assert run(col("s").ilike("PEAR%"), batch) == [False, True, None, False]


# ── temporal ────────────────────────────────────────────────────────────────


@pytest.fixture
def dates():
    """1970-01-01 and 2024-03-15 as ``date32``.

    Built through PyArrow and the C Data Interface rather than
    ``ma.array(..., type=date32())``: the Python `array()` builder has no
    temporal path (`unsupported type: date32`), which is a gap in the *array*
    bindings, not in the expression layer under test here.
    """
    import datetime

    return ma.record_batch(
        pa.record_batch(
            {"d": pa.array([datetime.date(1970, 1, 1), datetime.date(2024, 3, 15)])}
        )
    )


def test_temporal_extraction(dates):
    assert run(col("d").year(), dates) == [1970, 2024]
    assert run(col("d").month(), dates) == [1, 3]
    assert run(col("d").day(), dates) == [1, 15]
    assert run(col("d").quarter(), dates) == [1, 1]


def test_date_trunc_keeps_the_input_type(dates):
    truncated = col("d").date_trunc("year").execute(dates)
    assert str(truncated.type()) == "date32"


def test_date_trunc_rejects_a_bad_unit_when_the_expression_is_built():
    """Not on the first row that evaluates it — the unit is parsed at
    construction."""
    with pytest.raises(Exception, match="fortnight"):
        col("d").date_trunc("fortnight")


# ── nulls, conditionals, membership, casting ────────────────────────────────


def test_null_predicates(batch):
    assert run(col("a").is_null(), batch) == [False, False, True, False]
    assert run(col("a").is_valid(), batch) == [True, True, False, True]


def test_is_nan_is_null_where_the_input_is_null():
    """A null is not a NaN — that is the difference from ``is_null``."""
    b = ma.record_batch({"x": ma.array([1.0, float("nan"), None], type=ma.float64())})
    assert run(col("x").is_nan(), b) == [False, True, None]


def test_fill_null_and_nullif(batch):
    assert run(col("a").fill_null(lit(0)), batch) == [1, 2, 0, 4]
    assert run(col("a").nullif(lit(2)), batch) == [1, None, None, 4]


def test_conditionals_unify_a_mixed_numeric_pair(batch):
    """``lit(0)`` infers ``int64``, so filling a float column with it would hit
    ``FillNullKernel``'s ``expect_same_dtype`` unless the branches are unified
    first. This is the shape a Python caller writes without thinking."""
    b = ma.record_batch({"f": ma.array([1.5, None, 3.5], type=ma.float64())})
    assert run(col("f").fill_null(0), b) == [1.5, 0.0, 3.5]
    assert run(ma.coalesce(col("f"), lit(0)), b) == [1.5, 0.0, 3.5]
    assert run(if_else(col("f").is_null(), 0, col("f")), b) == [1.5, 0.0, 3.5]


def test_coalesce_of_incompatible_types_raises(batch):
    """No common type, so the guess is not made — the kernel names both."""
    with pytest.raises(Exception):
        run(ma.coalesce(col("s"), col("b")), batch)


def test_coalesce_takes_the_first_valid(batch):
    assert run(ma.coalesce(col("a"), col("b")), batch) == [1, 2, 30, 4]
    assert run(col("a").coalesce(col("b")), batch) == [1, 2, 30, 4]


def test_coalesce_needs_an_argument():
    with pytest.raises(ValueError):
        ma.coalesce()


def test_if_else_treats_a_null_condition_as_false(batch):
    """Arrow's ``ExecArrayCaseWhen`` rule, and PyArrow's ``pc.case_when``."""
    assert run(if_else(col("a") > 1, lit(1), lit(0)), batch) == [0, 1, 0, 1]


def test_case_when_multi_branch(batch):
    expr = ma.case_when(
        (col("b") < 15, lit("low")),
        (col("b") < 35, lit("mid")),
        else_=lit("high"),
    )
    assert run(expr, batch) == ["low", "mid", "mid", "high"]


def test_case_when_without_an_else_is_null(batch):
    expr = ma.case_when((col("b") < 15, lit("low")))
    assert run(expr, batch) == ["low", None, None, None]


def test_case_when_needs_a_branch():
    with pytest.raises(ValueError):
        ma.case_when()


def test_isin_against_a_list_and_an_array(batch):
    assert run(col("b").isin([20, 40]), batch) == [False, True, False, True]
    assert run(col("b").isin(ma.array([20, 40])), batch) == [
        False,
        True,
        False,
        True,
    ]


def test_cast_changes_the_type(batch):
    # Compared by name: the bound `DataType` registers no `__eq__`, so
    # `ma.int32() == ma.int32()` is an identity test and answers False.
    # The rest of the Python suite compares dtypes the same way.
    assert str(col("b").cast(ma.int32()).execute(batch).type()) == "int32"


def test_unsafe_cast_nulls_an_unparseable_string():
    b = ma.record_batch({"s": ma.array(["1", "nope"])})
    assert run(col("s").cast(ma.int64(), safe=False), b) == [1, None]


def test_array_length_over_a_list_column():
    b = ma.record_batch(
        {"xs": ma.array([[1, 2, 3], [], None], type=ma.list_(ma.int64()))}
    )
    assert run(col("xs").array_length(), b) == [3, 0, None]


# ── aggregates ──────────────────────────────────────────────────────────────


def test_reductions_name_themselves_after_the_function():
    assert col("a").sum().name() == "sum"
    assert col("a").mean().name() == "mean"


def test_alias_names_the_output_column():
    assert col("a").sum().alias("total").name() == "total"
    assert col("a").sum(alias="total").name() == "total"


def test_aggregate_by_name_matches_the_dedicated_verb():
    assert col("a").aggregate("sum").render() == col("a").sum().render()


def test_an_unknown_aggregate_raises():
    with pytest.raises(Exception, match="frobnicate"):
        col("a").aggregate("frobnicate")


def test_count_star_is_not_count_of_a_column():
    """``count(x)`` counts non-null values; ``count(*)`` counts rows. The two
    differ on any nullable column, which is why one is a free function."""
    assert count_star().name() == "count_star"
    assert count_star().referenced_columns() == []
    assert col("a").count().referenced_columns() == ["a"]


def test_aggregate_repr():
    assert repr(col("a").sum().alias("t")).startswith("<marrow.Aggregate:")
