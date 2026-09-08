"""The corpus: case format, transpile rules, and the SQL lane's verdicts."""

import pyarrow as pa
import pytest

from devkit.golden import (
    PREFIX,
    Case,
    CaseFormat,
    ExpectedTable,
    PythonTranspiler,
    SqlLane,
)


def make_case(
    sql="SELECT 1",
    expected=None,
    skips=(),
    body="return table('basic')",
    name="test_golden_x",
):
    return Case(
        path=None,
        name=name,
        sql=sql,
        prose="",
        expected=expected if expected is not None else pa.table({"a": [1]}),
        imports="",
        body=body,
        skips=set(skips),
        xfail=None,
    )


# ---------------------------------------------------------------------------
# The typed-TSV expectation block
# ---------------------------------------------------------------------------


def test_the_expectation_block_round_trips():
    """Text in, typed Arrow out, and back to the same text."""
    table = pa.table(
        {
            "name": pa.array(["a", None], pa.string()),
            "n": pa.array([1, 2], pa.int64()),
            "ok": pa.array([True, False], pa.bool_()),
        }
    )
    rendered = ExpectedTable.render(table)
    assert rendered.splitlines()[0] == "name:string\tn:int64\tok:bool"
    assert ExpectedTable.parse(rendered.splitlines(), "case").equals(table)


def test_strings_are_quoted_because_mojo_format_strips_trailing_space():
    """The `words` fixture holds `"  pad  "`.

    Unquoted, `mojo format` ate the trailing spaces inside the docstring and
    the expectation silently became a different string.
    """
    assert ExpectedTable.render_value("  pad  ") == "'  pad  '"
    assert ExpectedTable.render_value(None) == ExpectedTable.NULL


def test_an_unmapped_type_is_refused_rather_than_cast():
    with pytest.raises(SystemExit, match="unmapped arrow type"):
        ExpectedTable.type_name(pa.decimal128(10, 2))


def test_a_row_that_does_not_match_the_header_is_refused():
    with pytest.raises(SystemExit, match="row has"):
        ExpectedTable.parse(["a:int64\tb:int64", "1"], "case")


# ---------------------------------------------------------------------------
# Mojo -> Python
# ---------------------------------------------------------------------------


def transpile(body):
    return PythonTranspiler.transpile(make_case(body=body))


def test_the_transfer_sigil_is_dropped_in_argument_position():
    """`join(right^, ...)` reads as an XOR with a missing operand in Python.

    That is a SyntaxError, and it aborts collection for the *whole* corpus --
    not just the five join cases carrying it.
    """
    assert "right^" not in transpile("return left.join(right^, [0], [0])")
    assert "left.join(right, [0], [0])" in transpile(
        "return left.join(right^, [0], [0])"
    )


def test_the_xor_operator_and_a_caret_in_a_string_survive():
    """The rule is narrow on purpose: both of these must be left alone."""
    assert "^" in transpile('return col("p", bool_) ^ col("q", bool_)')
    assert '"^[a-z]"' in transpile('return regexp_matches(s, "^[a-z]")')


def test_var_is_dropped_and_ddof_becomes_a_verb():
    # `"var x = 1".endswith("x = 1")` is True, so an `endswith` here asserts
    # nothing: the rule could be deleted, or inverted to *add* `var`, and the
    # test would still pass.  A surviving `var` is a SyntaxError that aborts
    # collection for the whole corpus.
    body = transpile("var x = 1\nvar y = x")
    assert "var " not in body
    assert "x = 1" in body and "y = x" in body
    assert ".stddev_samp()" in transpile("return c.stddev[1]()")
    assert ".variance()" in transpile("return c.variance[0]()")
    assert ".var_samp()" in transpile("return c.variance[1]()")


def test_the_signature_is_written_not_transcribed():
    """`raises` and `-> DynRelation` must never reach Python."""
    rendered = transpile("return table('basic')")
    assert rendered.startswith("def plan():")
    assert "raises" not in rendered and "DynRelation" not in rendered


# ---------------------------------------------------------------------------
# One mismatch dump for every lane
# ---------------------------------------------------------------------------


def test_the_mismatch_dump_puts_each_table_under_its_own_heading():
    """Both lanes report through this, so a swap mislabels every failure.

    Asserting only that the two labels appear would pass with the tables
    exchanged, or with no tables at all.
    """
    case = make_case(expected=pa.table({"only_expected": [1]}))
    dump = case.mismatch(pa.table({"only_actual": [2]}))

    expected_at = dump.index("expected (duckdb)")
    actual_at = dump.index("actual (marrow)")
    assert expected_at < actual_at
    # Each table has to sit under its own heading, not merely be present.
    assert "only_expected" in dump[expected_at:actual_at]
    assert "only_actual" in dump[actual_at:]


# ---------------------------------------------------------------------------
# The SQL lane's four verdicts
# ---------------------------------------------------------------------------


class FakeLane(SqlLane):
    """A lane whose `run` answers with whatever the test wants."""

    def __init__(self, answer):
        super().__init__(corpus=None)
        self._answer = answer

    def run(self, sql):
        if isinstance(self._answer, Exception):
            raise self._answer
        return self._answer


class CountingLane(SqlLane):
    """A lane whose verdict for each case is dictated by the test."""

    def __init__(self, verdicts):
        super().__init__(corpus=None)
        self._verdicts = verdicts

    def outcome(self, case):
        return self._verdicts[case.name]


def counting_corpus(verdicts):
    """A lane over cases named `test_golden_<stem>`, one per verdict."""

    class FakeCorpus:
        def cases(self):
            return [make_case(name=PREFIX + stem) for stem in verdicts]

    lane = CountingLane({PREFIX + stem: v for stem, v in verdicts.items()})
    lane.corpus = FakeCorpus()
    return lane


def test_coverage_tallies_each_verdict_separately():
    """The printed table is this tally; a status folded into another is
    invisible unless the counts are asserted."""
    lane = counting_corpus(
        {
            "a": ("pass", ""),
            "b": ("pass", ""),
            "c": ("declined", "sql: unsupported function 'X'"),
            "d": ("declined", "sql: unsupported function 'X'"),
            "e": ("declined", "sql: expected ')'"),
            "f": ("divergent", "both lanes skip it"),
            "g": ("wrong", "nope"),
        }
    )
    counts, reasons, wrong = lane.coverage()
    assert dict(counts) == {"pass": 2, "declined": 3, "divergent": 1, "wrong": 1}
    # Grouped by reason, so the table shows kinds of gap rather than one line
    # per case.
    assert reasons["sql: unsupported function 'X'"] == 2
    assert reasons["sql: expected ')'"] == 1
    assert wrong == ["g"]


def test_the_report_returns_exactly_the_cases_it_lists():
    """The caller gates on this, so it must not be a second count of the same
    fact -- a summary once claimed "0 wrong" while listing every case."""
    lane = counting_corpus({"a": ("pass", ""), "b": ("wrong", "nope")})
    printed = []
    wrong = lane.report(echo=printed.append)
    body = "\n".join(printed)
    assert wrong == ["b"]
    assert "  wrong:        1" in body
    assert "        b" in body


def test_the_recorded_answer_is_a_pass():
    expected = pa.table({"a": [1]})
    assert FakeLane(expected).outcome(make_case(expected=expected))[0] == "pass"


def test_a_refusal_is_declined():
    """The front end recognised the query and said no -- a grammar gap."""
    status, detail = FakeLane(Exception("sql: unsupported function 'ARG_MIN'")).outcome(
        make_case()
    )
    assert status == "declined"
    assert detail == "sql: unsupported function 'ARG_MIN'"


def test_an_error_without_the_marker_is_wrong_not_declined():
    """A planner defect escaping from inside is not a grammar gap.

    Counting these as skips is how five real bugs once hid in the declined
    column.
    """
    status, detail = FakeLane(KeyError("column '' not found in schema")).outcome(
        make_case()
    )
    assert status == "wrong"
    assert "KeyError" in detail


def test_a_different_answer_is_wrong():
    status, detail = FakeLane(pa.table({"a": [2]})).outcome(
        make_case(expected=pa.table({"a": [1]}))
    )
    assert status == "wrong"
    assert "expected (duckdb)" in detail


def test_different_columns_say_so_rather_than_dumping_both_tables():
    status, detail = FakeLane(pa.table({"b": [1]})).outcome(
        make_case(expected=pa.table({"a": [1]}))
    )
    assert status == "wrong"
    assert "columns ['b'] != ['a']" in detail


def test_a_case_both_other_lanes_skip_is_divergent_not_wrong():
    """The corpus already judged it unassertable, so a mismatch here is
    reporting someone else's documented decision."""
    status, _ = FakeLane(pa.table({"a": [2]})).outcome(
        make_case(expected=pa.table({"a": [1]}), skips=("mojo", "python"))
    )
    assert status == "divergent"


def test_answering_a_case_the_others_skip_still_counts_as_a_pass():
    """That is a real capability the other lanes lack."""
    expected = pa.table({"a": [1]})
    status, _ = FakeLane(expected).outcome(
        make_case(expected=expected, skips=("mojo", "python"))
    )
    assert status == "pass"


# ---------------------------------------------------------------------------
# The case file format
# ---------------------------------------------------------------------------


def test_a_case_without_an_expected_block_is_refused(tmp_path):
    path = tmp_path / "x.mojo"
    path.write_text(
        f'{CaseFormat.DEF}\n    """\n    SELECT 1\n    """\n    return table("basic")\n'
    )
    with pytest.raises(SystemExit, match="no `-- expected` block"):
        CaseFormat.parse(path)


def test_the_markers_are_read_off_the_docstring(tmp_path):
    path = tmp_path / "x.mojo"
    path.write_text(
        f"{CaseFormat.DEF}\n"
        '    """\n'
        "    SELECT 1\n"
        "\n"
        "    -- skip mojo\n"
        "    -- xfail float keys collapse\n"
        "    -- expected\n"
        "    a:int64\n"
        "    1\n"
        '    """\n'
        '    return table("basic")\n'
    )
    case = CaseFormat.parse(path)
    assert case.sql == "SELECT 1"
    assert case.skipped("mojo") and not case.skipped("python")
    assert case.xfail == "float keys collapse"
    assert case.expected.equals(pa.table({"a": pa.array([1], pa.int64())}))


def test_the_sql_lane_runs_with_the_corpus_thread_budget(monkeypatch):
    """`--num-threads` reaches the query, not just the corpus object.

    Hardcoding it would make the option silently inert for this lane while the
    singleton test still passed.
    """
    import sys
    import types

    seen = []

    class FakePlan:
        def to_pyarrow(self, num_threads):
            seen.append(num_threads)
            return pa.table({"a": [1]})

    fake = types.ModuleType("marrow")
    fake.sql = lambda sql, tables: FakePlan()
    fake.read_ipc_file = lambda path: [object()]
    monkeypatch.setitem(sys.modules, "marrow", fake)

    class FakeCorpus:
        num_threads = 5
        fixtures = types.SimpleNamespace(path=lambda name: name)

    lane = SqlLane(FakeCorpus())
    lane.run("SELECT 1")
    assert seen == [5]


def test_dedent_strips_the_indent_and_nothing_else():
    """`textwrap.dedent` is unusable here: the expectation block is
    tab-separated, and a line whose data begins with a tab would defeat a
    common-prefix calculation.  Stripping more than the four spaces corrupts it.
    """
    assert CaseFormat.dedent("    a:int64\tb:int64") == "a:int64\tb:int64"
    # A data line that itself starts with a tab keeps that tab.
    assert CaseFormat.dedent("    \tvalue") == "\tvalue"
    # Deeper indentation keeps its remainder.
    assert CaseFormat.dedent("        nested") == "    nested"
    # A blank line stays blank rather than becoming whitespace.
    assert CaseFormat.dedent("   ") == ""


# ---------------------------------------------------------------------------
# The corpus and its derived artefacts
# ---------------------------------------------------------------------------


def scratch_corpus(tmp_path, cases):
    """A corpus over a scratch checkout holding *cases* as `{stem: source}`."""
    from devkit.golden import Corpus
    from devkit.mojo import Repo

    directory = tmp_path / "golden" / "cases"
    directory.mkdir(parents=True)
    for stem, source in cases.items():
        (directory / f"{stem}.mojo").write_text(source)
    return Corpus(Repo(tmp_path))


def case_source(sql="SELECT 1", markers=(), body='    return table("basic")'):
    lines = [CaseFormat.DEF, '    """', f"    {sql}", ""]
    lines += [f"    {marker}" for marker in markers]
    lines += ["    -- expected", "    a:int64", "    1", '    """', body]
    return "\n".join(lines) + "\n"


def test_the_generated_module_omits_a_case_the_mojo_lane_skips(tmp_path):
    """`-- skip mojo` means the case does not compile.

    The whole selection is one compilation unit, so including one would fail
    every case in the run.
    """
    from devkit.golden import MojoCodegen

    corpus = scratch_corpus(
        tmp_path,
        {
            "kept": case_source(),
            "dropped": case_source(markers=["-- skip mojo"]),
        },
    )
    rendered = MojoCodegen(corpus.directory.name).render(corpus.cases())
    assert "test_golden_kept" in rendered
    assert "dropped" not in rendered


def test_the_generated_module_imports_through_the_corpus_directory(tmp_path):
    from devkit.golden import MojoCodegen

    corpus = scratch_corpus(tmp_path, {"kept": case_source()})
    rendered = MojoCodegen(corpus.directory.name).render(corpus.cases())
    assert "from golden.cases.kept import plan as _kept" in rendered
    assert "from golden.helpers import check" in rendered
    assert 'check("test_golden_kept", _kept())' in rendered


def test_an_empty_result_is_written_as_one_empty_batch(tmp_path):
    """`write_table` emits *no* batch for a zero-row table.

    The Mojo lane would then read a file with 0 batches and could not tell an
    empty result from a missing expectation.
    """
    import pyarrow.ipc

    source = "\n".join(
        [
            CaseFormat.DEF,
            '    """',
            "    SELECT 1 WHERE FALSE",
            "",
            "    -- expected",
            "    a:int64",
            '    """',
            '    return table("basic")',
        ]
    )
    corpus = scratch_corpus(tmp_path, {"empty": source + "\n"})
    corpus.write_expectations()

    path = corpus.expectation_dir / "test_golden_empty.arrow"
    with pyarrow.ipc.open_file(path) as reader:
        assert reader.num_record_batches == 1
        assert reader.get_batch(0).num_rows == 0


def test_prepare_writes_every_artefact_collection_depends_on(tmp_path):
    corpus = scratch_corpus(tmp_path, {"kept": case_source()})
    assert corpus.prepare()
    assert corpus.generated.exists()
    assert (corpus.expectation_dir / "test_golden_kept.arrow").exists()
    assert corpus.fixtures.path("basic").exists()


def test_xfail_reasons_are_keyed_by_the_shared_item_name(tmp_path):
    """Both lanes name an item identically, which is how one marker reaches both."""
    corpus = scratch_corpus(
        tmp_path,
        {
            "broken": case_source(markers=["-- xfail float keys collapse"]),
            "fine": case_source(),
        },
    )
    assert corpus.xfail_reasons() == {"test_golden_broken": "float keys collapse"}


def test_the_corpus_is_one_object_per_process():
    """`--num-threads` is set on the corpus and read back off it.

    Two instances would silently drop the setting, with nothing red anywhere.
    """
    from devkit.golden import corpus

    corpus().num_threads = 7
    try:
        assert corpus() is corpus()
        assert corpus().num_threads == 7
    finally:
        corpus().num_threads = 0


# ---------------------------------------------------------------------------
# Installing the Python lane
# ---------------------------------------------------------------------------


def install(corpus, namespace=None):
    recorded = {}
    target = {}
    corpus.install_python_lane(
        namespace if namespace is not None else {"table": lambda name: name},
        lambda name, plan: recorded.setdefault(name, plan),
        target,
    )
    return target, recorded


def test_each_case_becomes_a_function_named_like_the_mojo_item(tmp_path):
    """`-k` has to select the same cases in both lanes."""
    corpus = scratch_corpus(tmp_path, {"kept": case_source()})
    target, _ = install(corpus)
    assert set(target) == {"test_golden_kept"}
    assert target["test_golden_kept"].__name__ == "test_golden_kept"


def test_a_case_the_python_lane_skips_is_marked_skip(tmp_path):
    corpus = scratch_corpus(
        tmp_path,
        {
            "kept": case_source(),
            "dropped": case_source(markers=["-- skip python"]),
        },
    )
    target, _ = install(corpus)
    marks = [m.name for m in target["test_golden_dropped"].pytestmark]
    assert "skip" in marks
    assert not getattr(target["test_golden_kept"], "pytestmark", [])


def test_a_case_body_cannot_leak_into_the_shared_vocabulary(tmp_path):
    """Each case executes in its own scope, or one case's `plan` becomes
    every later case's."""
    corpus = scratch_corpus(tmp_path, {"kept": case_source()})
    namespace = {"table": lambda name: name}
    install(corpus, namespace)
    assert "plan" not in namespace
