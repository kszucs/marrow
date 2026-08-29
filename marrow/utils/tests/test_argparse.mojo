"""`ArgumentParser` — the generic argv handling lifted out of the previous
expression layer's parameter module, before that layer was deleted.

The cases here pin the grammar itself: which tokens are options, how a value
attaches to one, what is required, and what `--help` renders. They deliberately
cover the two halves the old `parse_params`/`split_cli_args` pair had no test
for at all — the `=`-joined form and the `--` terminator — because those are
where an argv loop's off-by-one hides.

Every case drives `parse` from a `List[String]`, never from the real `argv`,
which is the property the module was factored for.
"""

from std.testing import assert_equal, assert_false, assert_raises, assert_true

from ..argparse import ArgumentParser, parse_bool


def _parser() raises -> ArgumentParser:
    """A parser with one of each kind, reused by the cases that only care
    about one of them."""
    var p = ArgumentParser("query", description=String("Run a plan."))
    p.option("min-a", metavar=String("N"), help=String("lower bound"))
    p.option("limit", default=String("10"), help=String("row cap"))
    p.flag("verbose", short=String("v"), help=String("chatter"))
    p.positional("src", help=String("input path"))
    return p^


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


def test_argparse_binds_a_required_option() raises:
    var p = ArgumentParser("q")
    p.option("min-a")
    var args = p.parse(["--min-a", "5"])
    assert_equal(args.get("min-a"), String("5"))
    assert_true(args.supplied("min-a"))


def test_argparse_applies_a_default_when_absent() raises:
    var p = ArgumentParser("q")
    p.option("limit", default=String("10"))
    var args = p.parse(List[String]())
    assert_equal(args.get("limit"), String("10"))
    assert_false(args.supplied("limit"))


def test_argparse_a_supplied_value_beats_the_default() raises:
    var p = ArgumentParser("q")
    p.option("limit", default=String("10"))
    var args = p.parse(["--limit", "3"])
    assert_equal(args.get("limit"), String("3"))
    assert_true(args.supplied("limit"))


def test_argparse_equals_joined_and_space_separated_agree() raises:
    """`--name=value` and `--name value` are the same argument, and a value
    containing `=` survives the first split."""
    var p = ArgumentParser("q")
    p.option("expr")
    assert_equal(p.parse(["--expr", "a=b=c"]).get("expr"), String("a=b=c"))
    assert_equal(p.parse(["--expr=a=b=c"]).get("expr"), String("a=b=c"))


def test_argparse_equals_form_can_supply_an_empty_value() raises:
    """`--name=` is an explicitly empty string, not a missing value — the one
    thing the space-separated form cannot spell."""
    var p = ArgumentParser("q")
    p.option("prefix")
    var args = p.parse(["--prefix="])
    assert_equal(args.get("prefix"), String())
    assert_true(args.supplied("prefix"))


def test_argparse_short_alias_takes_both_forms() raises:
    var p = ArgumentParser("q")
    p.option("output", short=String("o"))
    assert_equal(p.parse(["-o", "r.parquet"]).get("output"), "r.parquet")
    assert_equal(p.parse(["-o=r.parquet"]).get("output"), "r.parquet")


def test_argparse_takes_a_negative_number_as_a_value() raises:
    """The token after an option is its value unconditionally, so `-5` is a
    number here rather than an unrecognized option."""
    var p = ArgumentParser("q")
    p.option("min-a")
    assert_equal(p.parse(["--min-a", "-5"]).get_int("min-a"), -5)


def test_argparse_an_optional_option_may_have_no_default() raises:
    var p = ArgumentParser("q")
    p.option("tag", required=False)
    var args = p.parse(List[String]())
    assert_false(args.supplied("tag"))
    assert_false(Bool(args.get_optional("tag")))
    assert_equal(args.get_or("tag", String("none")), String("none"))
    with assert_raises():
        _ = args.get("tag")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_argparse_unknown_option_raises() raises:
    var p = _parser()
    with assert_raises():
        _ = p.parse(["--nope", "1", "in.parquet"])


def test_argparse_missing_required_option_raises() raises:
    var p = ArgumentParser("q")
    p.option("min-a")
    with assert_raises():
        _ = p.parse(List[String]())


def test_argparse_dangling_option_raises() raises:
    """A trailing `--min-a` with nothing after it."""
    var p = ArgumentParser("q")
    p.option("min-a")
    with assert_raises():
        _ = p.parse(["--min-a"])


def test_argparse_a_positional_name_is_not_an_option() raises:
    """`positional("src")` declares `src`, not `--src` — otherwise a program
    would silently accept two spellings of one argument."""
    var p = ArgumentParser("q")
    p.positional("src")
    with assert_raises():
        _ = p.parse(["--src", "in.parquet"])


def test_argparse_rejects_a_duplicate_declaration() raises:
    var p = ArgumentParser("q")
    p.option("min-a")
    with assert_raises():
        p.option("min-a")


def test_argparse_rejects_a_reused_short_alias() raises:
    var p = ArgumentParser("q")
    p.option("output", short=String("o"))
    with assert_raises():
        p.flag("overwrite", short=String("o"))


def test_argparse_rejects_a_second_trailing_argument() raises:
    var p = ArgumentParser("q")
    p.trailing("rest")
    with assert_raises():
        p.trailing("more")


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


def test_argparse_flag_is_absent_by_default() raises:
    var p = _parser()
    var args = p.parse(["--min-a", "5", "in.parquet"])
    assert_false(args.flag("verbose"))
    assert_false(args.supplied("verbose"))


def test_argparse_flag_is_set_by_long_or_short_form() raises:
    var p = _parser()
    assert_true(p.parse(["--min-a", "5", "--verbose", "in"]).flag("verbose"))
    assert_true(p.parse(["--min-a", "5", "-v", "in"]).flag("verbose"))


def test_argparse_flag_rejects_an_attached_value() raises:
    var p = _parser()
    with assert_raises():
        _ = p.parse(["--min-a", "5", "--verbose=true", "in"])


def test_argparse_flag_of_an_undeclared_name_raises() raises:
    """A misspelled flag must not read as "absent"."""
    var p = _parser()
    var args = p.parse(["--min-a", "5", "in"])
    with assert_raises():
        _ = args.flag("verbse")


def test_argparse_flag_and_value_tables_do_not_overlap() raises:
    var p = _parser()
    var args = p.parse(["--min-a", "5", "-v", "in"])
    with assert_raises():
        _ = args.get("verbose")
    with assert_raises():
        _ = args.flag("min-a")


def test_argparse_short_circuit_flag_skips_the_required_check() raises:
    """The `--describe` shape: a flag meaning "do not run the program" has to
    work without the arguments it is asking about."""
    var p = ArgumentParser("q")
    p.option("min-a")
    p.option("limit", default=String("10"))
    p.flag("describe", short_circuit=True)
    var args = p.parse(["--describe"])
    assert_true(args.flag("describe"))
    assert_false(args.help_requested)
    # defaults still resolve, which is what a --describe implementation reads
    assert_equal(args.get("limit"), String("10"))


def test_argparse_an_ordinary_flag_does_not_skip_it() raises:
    var p = ArgumentParser("q")
    p.option("min-a")
    p.flag("verbose")
    with assert_raises():
        _ = p.parse(["--verbose"])


# ---------------------------------------------------------------------------
# Positionals, trailing, and `--`
# ---------------------------------------------------------------------------


def test_argparse_positionals_match_in_declaration_order() raises:
    var p = ArgumentParser("q")
    p.positional("src")
    p.positional("dst")
    var args = p.parse(["a.parquet", "b.parquet"])
    assert_equal(args.get("src"), String("a.parquet"))
    assert_equal(args.get("dst"), String("b.parquet"))


def test_argparse_positionals_interleave_with_options() raises:
    var p = _parser()
    var args = p.parse(["in.parquet", "--min-a", "5", "-v"])
    assert_equal(args.get("src"), String("in.parquet"))
    assert_equal(args.get("min-a"), String("5"))
    assert_true(args.flag("verbose"))


def test_argparse_missing_required_positional_raises() raises:
    var p = ArgumentParser("q")
    p.positional("src")
    with assert_raises():
        _ = p.parse(List[String]())


def test_argparse_surplus_positional_raises() raises:
    var p = ArgumentParser("q")
    p.positional("src")
    with assert_raises():
        _ = p.parse(["a.parquet", "b.parquet"])


def test_argparse_trailing_collects_the_surplus() raises:
    var p = ArgumentParser("q")
    p.positional("src")
    p.trailing("rest")
    var args = p.parse(["a.parquet", "x", "y"])
    assert_equal(args.get("src"), String("a.parquet"))
    var rest = args.trailing()
    assert_equal(len(rest), 2)
    assert_equal(rest[0], String("x"))
    assert_equal(rest[1], String("y"))


def test_argparse_trailing_is_empty_when_undeclared() raises:
    var p = ArgumentParser("q")
    p.positional("src")
    assert_equal(len(p.parse(["a.parquet"]).trailing()), 0)


def test_argparse_double_dash_stops_option_parsing() raises:
    """Everything after `--` is positional, dashes and all — the escape hatch
    for a value that would otherwise read as an option."""
    var p = ArgumentParser("q")
    p.option("min-a")
    p.positional("src")
    p.trailing("rest")
    var args = p.parse(["--min-a", "5", "--", "--not-a-flag", "-x"])
    assert_equal(args.get("min-a"), String("5"))
    assert_equal(args.get("src"), String("--not-a-flag"))
    var rest = args.trailing()
    assert_equal(len(rest), 1)
    assert_equal(rest[0], String("-x"))


def test_argparse_double_dash_is_not_itself_a_positional() raises:
    var p = ArgumentParser("q")
    p.positional("src")
    assert_equal(p.parse(["--", "a.parquet"]).get("src"), String("a.parquet"))


def test_argparse_lone_dash_is_a_positional() raises:
    """The conventional stdin spelling."""
    var p = ArgumentParser("q")
    p.positional("src")
    assert_equal(p.parse(["-"]).get("src"), String("-"))


# ---------------------------------------------------------------------------
# Typed accessors
# ---------------------------------------------------------------------------


def test_argparse_typed_accessors_convert() raises:
    var p = ArgumentParser("q")
    p.option("n")
    p.option("ratio")
    p.option("strict")
    var args = p.parse(["--n", "42", "--ratio", "0.25", "--strict", "TRUE"])
    assert_equal(args.get_int("n"), 42)
    assert_true(args.get_float("ratio") == Float64(0.25))
    assert_true(args.get_bool("strict"))


def test_argparse_typed_accessors_name_the_argument() raises:
    var p = ArgumentParser("q")
    p.option("n")
    var args = p.parse(["--n", "twelve"])
    with assert_raises(contains="n"):
        _ = args.get_int("n")


def test_argparse_parse_bool_accepts_four_spellings() raises:
    assert_true(parse_bool("true"))
    assert_true(parse_bool("TRUE"))
    assert_true(parse_bool("1"))
    assert_false(parse_bool("false"))
    assert_false(parse_bool("False"))
    assert_false(parse_bool("0"))
    with assert_raises():
        _ = parse_bool("yes")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_argparse_usage_names_the_program_and_every_argument() raises:
    var p = _parser()
    var usage = p.usage()
    assert_true(usage.startswith("usage: query"))
    assert_true("--min-a N" in usage)
    assert_true("[--limit VALUE]" in usage)
    assert_true("[--verbose]" in usage)
    assert_true("<src>" in usage)


def test_argparse_usage_brackets_only_the_optional_arguments() raises:
    var p = _parser()
    var usage = p.usage()
    assert_true("[--min-a" not in usage)
    assert_true("[<src>]" not in usage)


def test_argparse_help_text_documents_every_argument() raises:
    var p = _parser()
    var text = p.help_text()
    assert_true(text.startswith("usage: query"))
    assert_true("Run a plan." in text)
    assert_true("options:" in text)
    assert_true("arguments:" in text)
    assert_true("-h, --help" in text)
    assert_true("lower bound (required)" in text)
    assert_true("row cap (default: 10)" in text)
    assert_true("-v, --verbose" in text)
    assert_true("chatter" in text)
    assert_true("input path (required)" in text)


def test_argparse_help_text_drops_the_builtin_when_shadowed() raises:
    var p = ArgumentParser("q")
    p.flag("help", short_circuit=True, help=String("my own help"))
    var text = p.help_text()
    assert_true("my own help" in text)
    assert_true("show this help message" not in text)
    assert_true("[-h]" not in p.usage())


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------


def test_argparse_help_is_recognized_without_declaration() raises:
    var p = _parser()
    assert_true(p.parse(["--help"]).help_requested)
    assert_true(p.parse(["-h"]).help_requested)
    assert_false(p.parse(["--min-a", "5", "in"]).help_requested)


def test_argparse_help_skips_the_required_check() raises:
    """`query --help` must print the page, not complain about the arguments
    the user is asking about."""
    var p = _parser()
    var args = p.parse(["--help"])
    assert_true(args.help_requested)
    assert_equal(args.get("limit"), String("10"))


def test_argparse_a_declared_help_shadows_the_builtin() raises:
    var p = ArgumentParser("q")
    p.flag("help", short_circuit=True)
    var args = p.parse(["--help"])
    assert_false(args.help_requested)
    assert_true(args.flag("help"))
