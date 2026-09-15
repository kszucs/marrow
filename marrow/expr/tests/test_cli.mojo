"""`marrow.expr.cli` — the renderers, and the command line a plan derives.

`QueryCli.run()` reads the process's real `argv` and can `exit()`, so it is not
reachable from a test driver. What it delegates to is: the two renderers, the
`--help` a plan's parameters produce, `parse` over a token list, and the
bindings those tokens become. The built `benchmarks/binary_size/query_cli` is
run end to end by `python/marrow/tests/test_compile.py`.
"""

from std.testing import assert_equal, assert_raises, assert_true

from ...builders import array
from ...dtypes import (
    bool_,
    date32,
    float64,
    int64,
    large_string,
    string,
    uint8,
)
from ...scalars import LargeStringScalar
from ...tabular import RecordBatch, record_batch
from ..builders import col, param, table
from ..cli import QueryCli, render_csv, render_table
from ..logical import DynRelation, DynValue


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


# ---------------------------------------------------------------------------
# QueryCli — the command line a plan derives
# ---------------------------------------------------------------------------


def _report_plan() raises -> DynRelation:
    return table(_batch()).filter(
        (
            col("id", int64)
            >= param("min-id", int64, default=Int64(0), help="lower bound")
        )
        & (col("id", int64) <= param("max-id", int64, help="upper bound"))
    )


def test_query_cli_help_lists_every_parameter_and_builtin() raises:
    var cli = QueryCli(_report_plan(), description="A report.")
    var text = cli.help_text()
    var wanted: List[String] = [
        "A report.",
        "--min-id int64",
        "lower bound",
        "(default: 0)",
        "--max-id int64",
        "upper bound",
        "(required)",
        "--describe",
        "-o, --output",
        "--format",
        "--max-rows",
    ]
    for ref expected in wanted:
        assert_true(expected in text, "missing from --help: " + expected)


def test_query_cli_describe_needs_no_parameters() raises:
    var cli = QueryCli(_report_plan())
    var args = cli.parse(["--describe"])
    assert_true(args.flag("describe"))
    with assert_raises(contains="max-id"):
        _ = cli.parse([])


def test_query_cli_parses_typed_bindings() raises:
    var plan = table(_batch()).filter(
        (col("id", int64) >= param("lo", int64))
        & (col("id", int64).cast(float64, safe=False) < param("hi", float64))
        & (col("name", string) != param("skip", string))
        & param("keep", bool_)
    )
    var cli = QueryCli(plan.copy())
    var values = cli.bindings(
        cli.parse(
            ["--lo", "2", "--hi", "3.5", "--skip", "carol", "--keep", "true"]
        )
    )
    var out = plan.execute(bindings=values)
    assert_true(out.columns[0].as_int64() == array([2], int64))

    var widths = table(_batch()).project(
        ["u", "ls"], [param("u", uint8), param("ls", large_string)]
    )
    var typed = QueryCli(widths^)
    var bound = typed.bindings(typed.parse(["--u", "200", "--ls", "text"]))
    assert_equal(bound["u"].as_uint8().value(), UInt8(200))
    assert_true(bound["ls"].isa[LargeStringScalar]())


def test_query_cli_binds_only_what_argv_mentions() raises:
    var cli = QueryCli(_report_plan())
    var values = cli.bindings(cli.parse(["--max-id", "2"]))
    assert_true("max-id" in values)
    assert_true("min-id" not in values)


def test_query_cli_a_bad_token_names_its_flag() raises:
    var cli = QueryCli(_report_plan())
    with assert_raises(contains="--max-id"):
        _ = cli.bindings(cli.parse(["--max-id", "abc"]))


def test_query_cli_refuses_an_out_of_range_integer() raises:
    var cli = QueryCli(table(_batch()).project(["u"], [param("u", uint8)]))
    with assert_raises(contains="out of range"):
        _ = cli.bindings(cli.parse(["--u", "300"]))


def test_query_cli_refuses_a_parameter_it_cannot_read() raises:
    with assert_raises(contains="cannot be read from the command line"):
        _ = QueryCli(table(_batch()).project(["d"], [param("d", date32())]))

    var optional = QueryCli(
        table(_batch()).project(["d"], [param("d", date32(), default=Int32(0))])
    )
    with assert_raises(contains="cannot be set from the command line"):
        _ = optional.bindings(optional.parse(["--d", "5"]))


def test_query_cli_refuses_a_parameter_named_like_a_builtin() raises:
    with assert_raises(contains="'format' is already declared"):
        _ = QueryCli(table(_batch()).project(["f"], [param("format", int64)]))


def test_query_cli_rejects_an_unknown_format_at_parse_time() raises:
    var cli = QueryCli(_report_plan())
    with assert_raises(contains="--format"):
        _ = cli.parse(["--max-id", "1", "--format", "xml"])


def test_query_cli_binds_many_parameters() raises:
    """Twelve bindings grow the `Dict[String, DynScalar]` past its first
    allocation — the shape the List-of-Variant defect in CLAUDE.md corrupts."""
    var names = List[String]()
    var values = List[DynValue]()
    var tokens = List[String]()
    for i in range(12):
        var name = String("p") + String(i)
        names.append(name.copy())
        values.append(param(name.copy(), int64))
        tokens.append(String("--") + name)
        tokens.append(String(i * 10))
    var cli = QueryCli(table(_batch()).project(names^, values^))
    var bound = cli.bindings(cli.parse(tokens))
    assert_equal(len(bound), 12)
    for i in range(12):
        assert_equal(
            bound[String("p") + String(i)].as_int64().value(), Int64(i * 10)
        )
