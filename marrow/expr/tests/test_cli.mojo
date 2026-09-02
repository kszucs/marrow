"""`marrow.expr.cli` — the output renderers and the parameter surface.

`QueryCli.parse()` reads the process's real `argv` and `run()` can `exit()`, so
neither is reachable from a test driver; what *is* testable, and what carries
the behaviour a user sees, is everything either one delegates to — the two
renderers, and that a declared parameter yields a node the plan binds. The
end-to-end command line is exercised by building `benchmarks/binary_size/
query_cli.mojo` and running it.
"""

from std.testing import assert_equal, assert_true

from ...builders import array
from ...dtypes import float64, int64
from ...tabular import RecordBatch, record_batch
from ..cli import QueryCli, render_csv, render_table


def _batch() raises -> RecordBatch:
    return record_batch(
        [
            array([1, 2, 3], int64),
            array(["alice", "bob", "carol"]),
        ],
        names=["id", "name"],
    )


def test_render_table_aligns_columns() raises:
    var out = render_table(_batch())
    assert_equal(
        out,
        String(
            "id  name \n--  -----\n1   alice\n2   bob  \n3   carol\n(3 rows)"
        ),
    )


def test_render_table_truncates_and_says_so() raises:
    var out = render_table(_batch(), max_rows=2)
    assert_true("(2 of 3 rows; --max-rows 0 for all)" in out)
    assert_true("carol" not in out)


def test_render_table_max_rows_zero_shows_all() raises:
    var out = render_table(_batch(), max_rows=0)
    assert_true("carol" in out)
    assert_true(out.endswith("(3 rows)"))


def test_render_table_spells_nulls() raises:
    var batch = record_batch([array([1, None, 3], int64)], names=["v"])
    assert_equal(
        render_table(batch),
        String("v   \n----\n1   \nnull\n3   \n(3 rows)"),
    )


def test_render_csv_has_a_header_and_every_row() raises:
    assert_equal(
        render_csv(_batch()),
        String("id,name\n1,alice\n2,bob\n3,carol\n"),
    )


def test_render_csv_quotes_delimiters_and_quotes() raises:
    var batch = record_batch([array(["a,b", 'say "hi"', "plain"])], names=["s"])
    assert_equal(
        render_csv(batch),
        String('s\n"a,b"\n"say ""hi"""\nplain\n'),
    )


def test_render_csv_writes_a_null_as_an_empty_field() raises:
    var batch = record_batch([array([1, None, 3], int64)], names=["v"])
    assert_equal(render_csv(batch), String("v\n1\n\n3\n"))


def test_query_cli_declares_help_for_every_argument() raises:
    """`param` / `argument` / `option` / `flag` all reach `--help`.

    Also the only thing that instantiates `_coerce_param` for a *float* dtype:
    the coercion is monomorphic on purpose, so a branch no program declares is
    a branch nothing elaborates, and `query_cli`'s single `int64` parameter
    would leave the floating-point arm unbuilt.
    """
    var cli = QueryCli(String("report"), description=String("A report."))
    var lo = cli.param(
        String("min-amount"),
        int64,
        default=Int64(0),
        help=String("lower bound"),
    )
    var rate = cli.param(
        String("rate"), float64, help=String("a required float")
    )
    cli.argument(String("src"), help=String("input file"))
    cli.option(String("tag"), default=String("none"), help=String("a label"))
    cli.flag(String("dry-run"), short=String("n"), help=String("do nothing"))

    var text = cli.help_text()
    var wanted: List[String] = [
        "A report.",
        "--min-amount",
        "lower bound",
        "(default: 0)",
        "--rate",
        "a required float",
        "src",
        "input file",
        "--tag",
        "-n, --dry-run",
        "--describe",
        "-o, --output",
        "--format",
        "--max-rows",
    ]
    for ref expected in wanted:
        assert_true(expected in text, "missing from --help: " + expected)

    # The declarations are ordinary nodes: the plan sees `Param[T]`, not a
    # CLI object.
    assert_equal(lo.name(), String("min-amount"))
    assert_equal(rate.name(), String("rate"))


def test_query_cli_reading_before_parse_is_a_named_error() raises:
    var cli = QueryCli(String("report"))
    cli.argument(String("src"))
    var raised = String()
    try:
        _ = cli.get(String("src"))
    except e:
        raised = String(e)
    assert_true("call parse() before" in raised, raised)
