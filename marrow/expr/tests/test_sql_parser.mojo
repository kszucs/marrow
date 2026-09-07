"""Lexing and parsing cases for `expr/sql.mojo`.

Everything here stops at `Parser.parse`: no schema, no catalogue, no plan. The
planner's half is covered end-to-end by `golden/test_sql_cases.py`, which runs
the recorded SQL of all 278 golden cases, and by `python/marrow/tests/
test_sql.py`.

Structure is asserted through `render`, which prints a parsed expression as an
s-expression. Comparing one string is how a precedence question gets a
one-line answer: `a + b * 2 > 10` is either `(> (+ a (* b 2)) 10)` or it is
wrong, and the assertion says which.
"""

from std.testing import assert_equal, assert_false, assert_true

from ..sql import Ascii, Ast, NONE, Parser


def render(ast: Ast, i: Int) -> String:
    """A parsed expression as an s-expression.

    One generic arm covers every operator, because an operator's tag *is* the
    operator: `(> (+ a (* b 2)) 10)` falls out of printing `kind` followed by
    the operands. Only the leaves and the two nodes carrying a name of their
    own — a call and a cast — need a case.
    """
    if i == NONE:
        return String("_")
    ref n = ast.nodes[i]
    if n.kind == "column":
        return n.qualified_name()
    if n.kind == "int" or n.kind == "float":
        return n.text.copy()
    if n.kind == "string":
        return "'" + n.text + "'"
    if n.kind == "bool":
        return String("true") if n.value == 1 else String("false")
    if n.kind == "null":
        return String("null")
    if n.kind == "star":
        return String("*") if n.qualifier == "" else n.qualifier + ".*"
    if n.kind == "cast":
        return "(cast " + render(ast, n.a) + " " + n.text + ")"

    var out = "(" + (n.text if n.kind == "func" else n.kind)
    if n.distinct:
        out += " distinct"
    var operands = [n.a, n.b, n.c]
    for operand in operands:
        if operand != NONE:
            out += " " + render(ast, operand)
    for k in range(n.kids_len):
        out += " " + render(ast, ast.kid(n, k))
    return out + ")"


def where_of(sql: String) raises -> String:
    """The `WHERE` clause of `sql`, rendered. Raises if it did not parse."""
    var ast = Parser.parse(sql)
    assert_true(ast.ok(), "did not parse: " + ast.error)
    return render(ast, ast.select.predicate)


def first_item(sql: String) raises -> String:
    var ast = Parser.parse(sql)
    assert_true(ast.ok(), "did not parse: " + ast.error)
    return render(ast, ast.select.items[0].node)


# -- tokenizer --------------------------------------------------------------


def test_tokenize_splits_words_numbers_and_punctuation() raises:
    var toks = Parser.scan("SELECT a, 12 FROM t")
    assert_equal(len(toks), 7)  # six lexemes and the EOF
    assert_true(toks[0].is_word("SELECT"))
    assert_equal(toks[1].text, "a")
    assert_true(toks[2].is_punct(","))
    assert_equal(toks[3].kind, "number")
    assert_false(toks[3].is_float)


def test_tokenize_keywords_are_case_insensitive() raises:
    var toks = Parser.scan("sElEcT")
    assert_true(toks[0].is_word("SELECT"))
    assert_equal(toks[0].text, "sElEcT")
    assert_equal(toks[0].upper, "SELECT")


def test_tokenize_quoted_identifier_is_not_a_keyword() raises:
    """`"select"` names a column, however it is spelled."""
    var toks = Parser.scan('"select" "a""b"')
    assert_equal(toks[0].kind, "word")
    assert_false(toks[0].is_word("SELECT"))
    assert_equal(toks[1].text, 'a"b')


def test_tokenize_string_literal_doubles_its_quote() raises:
    var toks = Parser.scan("'it''s'")
    assert_equal(toks[0].kind, "string")
    assert_equal(toks[0].text, "it's")


def test_tokenize_floats_carry_their_exponent() raises:
    var toks = Parser.scan("1 2.5 3e2 4.5e-3")
    assert_false(toks[0].is_float)
    assert_true(toks[1].is_float)
    assert_true(toks[2].is_float)
    assert_equal(toks[3].text, "4.5e-3")


def test_tokenize_exponent_does_not_swallow_a_following_word() raises:
    """`1 END` must not lex as the float `1e`."""
    var toks = Parser.scan("1 end")
    assert_false(toks[0].is_float)
    assert_true(toks[1].is_word("END"))


def test_tokenize_skips_both_comment_forms() raises:
    var toks = Parser.scan("a /* b */ c -- d\ne")
    assert_equal(len(toks), 4)
    assert_equal(toks[2].text, "e")


def test_tokenize_reports_an_unknown_character() raises:
    var toks = Parser.scan("a # b")
    assert_equal(toks[1].kind, "error")


def test_ascii_upper_leaves_non_letters_alone() raises:
    assert_equal(Ascii.upper("a_1é"), "A_1é")


# -- expression precedence --------------------------------------------------


def test_arithmetic_binds_tighter_than_comparison() raises:
    assert_equal(
        where_of("SELECT * FROM t WHERE a + b * 2 > 10"),
        "(> (+ a (* b 2)) 10)",
    )


def test_and_binds_tighter_than_or() raises:
    assert_equal(
        where_of("SELECT * FROM t WHERE a OR b AND c"), "(or a (and b c))"
    )


def test_not_binds_looser_than_comparison() raises:
    assert_equal(where_of("SELECT * FROM t WHERE NOT a = b"), "(not (= a b))")


def test_not_binds_tighter_than_and() raises:
    assert_equal(
        where_of("SELECT * FROM t WHERE NOT a AND b"), "(and (not a) b)"
    )


def test_unary_minus_binds_tighter_than_multiplication() raises:
    assert_equal(first_item("SELECT -a * 2 FROM t"), "(* (neg a) 2)")


def test_arithmetic_is_left_associative() raises:
    assert_equal(first_item("SELECT a - b - c FROM t"), "(- (- a b) c)")


def test_parentheses_override_precedence() raises:
    assert_equal(first_item("SELECT (a + b) * c FROM t"), "(* (+ a b) c)")


def test_concat_binds_looser_than_arithmetic() raises:
    assert_equal(first_item("SELECT a || b + c FROM t"), "(|| a (+ b c))")


# -- word-shaped predicates -------------------------------------------------


def test_between_does_not_swallow_the_following_and() raises:
    """The bound is parsed above `AND`, so the conjunction after it survives."""
    assert_equal(
        where_of("SELECT * FROM t WHERE x BETWEEN 1 AND 5 AND y"),
        "(and (between x 1 5) y)",
    )


def test_not_between_is_one_predicate() raises:
    assert_equal(
        where_of("SELECT * FROM t WHERE x NOT BETWEEN 1 AND 5"),
        "(not_between x 1 5)",
    )


def test_in_list_keeps_every_item() raises:
    assert_equal(
        where_of("SELECT * FROM t WHERE x IN (1, 2, 3)"), "(in x 1 2 3)"
    )


def test_not_in_is_one_predicate() raises:
    assert_equal(where_of("SELECT * FROM t WHERE x NOT IN (1)"), "(not_in x 1)")


def test_is_null_and_is_not_null() raises:
    assert_equal(where_of("SELECT * FROM t WHERE x IS NULL"), "(is_null x)")
    assert_equal(
        where_of("SELECT * FROM t WHERE x IS NOT NULL"), "(is_not_null x)"
    )


def test_like_variants() raises:
    assert_equal(where_of("SELECT * FROM t WHERE s LIKE 'a%'"), "(like s 'a%')")
    assert_equal(
        where_of("SELECT * FROM t WHERE s NOT LIKE 'a%'"), "(not_like s 'a%')"
    )
    assert_equal(
        where_of("SELECT * FROM t WHERE s ILIKE 'a%'"), "(ilike s 'a%')"
    )


# -- other expression forms -------------------------------------------------


def test_cast_keeps_a_multi_word_type() raises:
    assert_equal(
        first_item("SELECT CAST(v AS DOUBLE PRECISION) FROM t"),
        "(cast v DOUBLE PRECISION)",
    )


def test_cast_keeps_type_parameters() raises:
    assert_equal(
        first_item("SELECT CAST(v AS DECIMAL(10,2)) FROM t"),
        "(cast v DECIMAL(10,2))",
    )


def test_searched_case() raises:
    assert_equal(
        first_item("SELECT CASE WHEN a THEN 1 ELSE 2 END FROM t"),
        "(case 2 a 1)",
    )


def test_simple_case_desugars_to_equality() raises:
    """`CASE k WHEN 'a'` becomes `k = 'a'`, so the planner sees one shape."""
    assert_equal(
        first_item("SELECT CASE k WHEN 'a' THEN 1 END FROM t"),
        "(case (= k 'a') 1)",
    )


def test_function_call_and_count_star() raises:
    assert_equal(first_item("SELECT SUM(v) FROM t"), "(SUM v)")
    assert_equal(first_item("SELECT COUNT(*) FROM t"), "(COUNT *)")
    assert_equal(
        first_item("SELECT COUNT(DISTINCT k) FROM t"), "(COUNT distinct k)"
    )


def test_qualified_column_and_qualified_star() raises:
    assert_equal(first_item("SELECT e.name FROM emp e"), "e.name")
    assert_equal(first_item("SELECT e.* FROM emp e"), "e.*")


# -- clauses ----------------------------------------------------------------


def test_select_list_aliases_both_spellings() raises:
    var ast = Parser.parse("SELECT a AS x, b y, c FROM t")
    assert_true(ast.ok(), ast.error)
    assert_equal(ast.select.items[0].alias_name, "x")
    assert_equal(ast.select.items[1].alias_name, "y")
    assert_equal(ast.select.items[2].alias_name, "")


def test_a_clause_keyword_is_not_read_as_an_alias() raises:
    """The reason `SELECT a FROM t` has one item and a table."""
    var ast = Parser.parse("SELECT a FROM t")
    assert_true(ast.ok(), ast.error)
    assert_equal(ast.select.items[0].alias_name, "")
    assert_equal(ast.select.table, "t")


def test_distinct_is_recorded() raises:
    var ast = Parser.parse("SELECT DISTINCT a FROM t")
    assert_true(ast.ok(), ast.error)
    assert_true(ast.select.distinct)


def test_group_by_having_order_limit_offset() raises:
    var ast = Parser.parse(
        "SELECT k, SUM(v) t FROM b GROUP BY k HAVING SUM(v) > 1"
        " ORDER BY t DESC, k ASC LIMIT 10 OFFSET 5"
    )
    assert_true(ast.ok(), ast.error)
    assert_equal(len(ast.select.group_by), 1)
    assert_true(ast.select.having != NONE)
    assert_equal(len(ast.select.order), 2)
    assert_false(ast.select.order[0].ascending)
    assert_true(ast.select.order[1].ascending)
    assert_equal(ast.select.limit, 10)
    assert_equal(ast.select.offset, 5)


def test_order_by_null_placement_defaults_follow_duckdb() raises:
    """NULLs last in **both** directions, verified against DuckDB 1.5.1.

    Deriving the default from the direction instead would be wrong for DESC,
    and would also manufacture a conflict between two keys — `Sort` carries
    one flag for all of them — for a query that never asked about nulls.
    """
    var ast = Parser.parse("SELECT a FROM t ORDER BY a ASC, b DESC")
    assert_true(ast.ok(), ast.error)
    assert_false(ast.select.order[0].nulls_first)
    assert_false(ast.select.order[1].nulls_first)
    assert_false(ast.select.order[0].nulls_explicit)
    assert_false(ast.select.order[1].nulls_explicit)


def test_order_by_records_that_nulls_was_written() raises:
    """Only an explicit clause can conflict, so the parser records which
    keys carried one."""
    var ast = Parser.parse("SELECT a FROM t ORDER BY a NULLS FIRST, b")
    assert_true(ast.ok(), ast.error)
    assert_true(ast.select.order[0].nulls_explicit)
    assert_false(ast.select.order[1].nulls_explicit)


def test_limit_rejects_a_float() raises:
    """`_parse_digits` drops non-digit bytes, so an unguarded `LIMIT 1.5`
    would silently become `LIMIT 15`."""
    var ast = Parser.parse("SELECT a FROM t LIMIT 1.5")
    assert_false(ast.ok())
    assert_true("whole number" in ast.error)


def test_order_by_explicit_null_placement() raises:
    var ast = Parser.parse("SELECT a FROM t ORDER BY a NULLS FIRST")
    assert_true(ast.ok(), ast.error)
    assert_true(ast.select.order[0].nulls_first)


def test_join_kinds_and_keys() raises:
    var ast = Parser.parse(
        "SELECT * FROM emp e LEFT JOIN dept d ON e.did = d.did AND e.x = d.y"
    )
    assert_true(ast.ok(), ast.error)
    assert_equal(len(ast.select.joins), 1)
    assert_equal(ast.select.joins[0].kind, "left")
    assert_equal(ast.select.joins[0].alias_name, "d")
    assert_equal(len(ast.select.joins[0].left_keys), 2)
    assert_equal(ast.select.joins[0].left_keys[0], "e.did")
    assert_equal(ast.select.joins[0].right_keys[1], "d.y")


def test_bare_join_is_inner() raises:
    var ast = Parser.parse("SELECT * FROM a JOIN b ON a.x = b.x")
    assert_true(ast.ok(), ast.error)
    assert_equal(ast.select.joins[0].kind, "inner")


# -- errors -----------------------------------------------------------------


def test_error_reports_the_first_problem_only() raises:
    var ast = Parser.parse("SELECT FROM WHERE")
    assert_false(ast.ok())


def test_error_on_truncated_where() raises:
    var ast = Parser.parse("SELECT a FROM t WHERE")
    assert_false(ast.ok())
    assert_true("unexpected" in ast.error)


def test_error_on_unbalanced_parenthesis() raises:
    var ast = Parser.parse("SELECT (a FROM t")
    assert_false(ast.ok())


def test_error_on_non_equi_join() raises:
    var ast = Parser.parse("SELECT * FROM a JOIN b ON a.x > b.x")
    assert_false(ast.ok())
    assert_true("equality" in ast.error)


def test_error_carries_an_offset() raises:
    var ast = Parser.parse("SELECT a # b FROM t")
    assert_false(ast.ok())
    assert_true(ast.error_pos > 0)


def test_trailing_tokens_are_an_error() raises:
    var ast = Parser.parse("SELECT a FROM t garbage here")
    assert_false(ast.ok())
