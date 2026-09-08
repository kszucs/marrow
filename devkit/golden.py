"""The cross-lane golden query corpus: fixtures, case format, codegen, transpile.

One case is one file, `golden/cases/<name>.mojo`, and it is Mojo source.  The
Mojo lane compiles it ahead of time; the Python lane runs the *same text*
through a mechanical transpile and the runtime bindings.  Keeping one spelling
is the point -- the two lanes drifted once already, when the AOT lane had no
boolean column leaf and the Python twin silently tested something else.

This module owns everything that is not the case vocabulary itself.  That stays
in `golden/`: `helpers.py` for Python and `helpers.mojo` for Mojo, beside the
cases they serve.

**`duckdb` is imported inside `Corpus.regenerate`, never at module scope.**  The
`dev` environment has no duckdb, and comparing against an expectation does not
need one -- only producing an expectation does.
"""

import ast
import functools
import re
import textwrap
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import pyarrow as pa

from .fixtures import TABLES, FixtureSet
from .mojo import Repo, write_if_changed

#: Every case name starts with this, so pytest's `-k` selects the same items in
#: both lanes and the Mojo runner can report by a name Python also knows.
PREFIX = "test_golden_"


@functools.cache
def corpus():
    """The corpus for this process.

    A singleton because `golden/helpers.py` is the case *vocabulary*:
    `table(name)` and `check(name, plan)` have to read the same in both lanes,
    and threading a handle through every case body is exactly what the
    vocabulary exists to avoid.  Execution knobs set from the command line ride
    on it for the same reason -- see `Corpus.num_threads`.
    """
    return Corpus(Repo.locate())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
# Fixtures are **files**, not construction code: all three consumers -- the AOT
# lane, the runtime lane, and the DuckDB twin producing the expectations -- read
# the same bytes. Building the table per lane would let them drift on exactly
# the thing under test.
#
# Nulls appear in every column that can hold one. A fixture without them tests
# the happy path of kernels whose null handling is the interesting part.


# ---------------------------------------------------------------------------
# The typed-TSV expectation block
# ---------------------------------------------------------------------------


class ExpectedTable:
    """The tab-separated block that records a case's expected result.

    A type per column is what lets the text be read back into a *typed* Arrow
    table.  sqllogictest's `query IIR` letters coerce results before comparing,
    which hides exactly the type bugs an Arrow engine should be asserting.
    """

    NULL = "NULL"

    #: Only the types the corpus uses are mapped; an unmapped type is an error
    #: rather than a silent cast.
    TYPES = {
        "string": pa.string(),
        "int64": pa.int64(),
        "int32": pa.int32(),
        "double": pa.float64(),
        "bool": pa.bool_(),
        "date32": pa.date32(),
        # Microseconds, which is what DuckDB's `TIMESTAMP` is and what the
        # `events` fixture holds. A unit belongs in the *value*, not the type
        # name: an expectation written as `timestamp` and read back as some
        # other unit would compare equal on the numbers while meaning different
        # instants.
        "timestamp": pa.timestamp("us"),
    }
    NAMES = {str(dtype): name for name, dtype in TYPES.items()}

    @classmethod
    def type_name(cls, dtype):
        try:
            return cls.NAMES[str(dtype)]
        except KeyError:
            raise SystemExit(
                f"golden: unmapped arrow type {dtype!r}; add it to ExpectedTable.TYPES"
            )

    @classmethod
    def render_value(cls, value):
        """One cell.  Strings are **quoted**, and that is load-bearing.

        `mojo format` strips trailing whitespace inside a docstring.  The
        `words` fixture holds `"  pad  "`, so an unquoted block lost the
        trailing spaces and the expectation silently became a different string
        -- a corpus that asserts the wrong answer.  Quoting keeps every line
        ending in a printable character.  It also lets string data contain a tab
        or a newline, which the bare format could not represent at all.
        """
        if value is None:
            return cls.NULL
        if isinstance(value, (date, datetime)):
            # ISO 8601, quoted like a string for the same reason. `datetime`
            # subclasses `date`, and each one's own `isoformat` is the right
            # spelling, so a single branch covers both. `isoformat` keeps
            # microseconds when there are any and omits the fractional part when
            # there are none, which round-trips exactly.
            return repr(value.isoformat())
        if isinstance(value, (str, float)):
            return repr(value)
        return str(value)

    @classmethod
    def parse_value(cls, text, dtype):
        if text == cls.NULL:
            return None
        if dtype == pa.bool_():
            return text == "True"
        value = ast.literal_eval(text)
        # The temporal cells arrive as quoted ISO strings, so the literal_eval
        # above yields the text and the constructor below yields the value.
        if dtype == pa.date32():
            return date.fromisoformat(value)
        if dtype == pa.timestamp("us"):
            return datetime.fromisoformat(value)
        return value

    @classmethod
    def render(cls, table):
        """A pyarrow table -> the block that goes in a docstring."""
        header = "\t".join(
            f"{field.name}:{cls.type_name(field.type)}" for field in table.schema
        )
        rows = [
            "\t".join(cls.render_value(value) for value in row.values())
            for row in table.to_pylist()
        ]
        return "\n".join([header, *rows])

    @classmethod
    def parse(cls, lines, where):
        """The block back into a typed pyarrow table."""
        if not lines:
            raise SystemExit(f"golden: {where}: empty `{CaseFormat.MARKER}` block")
        columns, types = [], []
        for spec in lines[0].split("\t"):
            column, _, name = spec.partition(":")
            if name not in cls.TYPES:
                raise SystemExit(f"golden: {where}: unknown type {name!r} in header")
            columns.append(column)
            types.append(cls.TYPES[name])
        rows = [line.split("\t") for line in lines[1:]]
        for row in rows:
            if len(row) != len(columns):
                raise SystemExit(
                    f"golden: {where}: row has {len(row)} fields, "
                    f"header has {len(columns)}"
                )
        return pa.table(
            {
                column: pa.array(
                    [cls.parse_value(row[index], types[index]) for row in rows],
                    types[index],
                )
                for index, column in enumerate(columns)
            }
        )


# ---------------------------------------------------------------------------
# The case-file format
# ---------------------------------------------------------------------------


class Case:
    """One `golden/cases/<stem>.mojo`.

    `sql` is the docstring's first paragraph, `prose` whatever follows it, and
    `expected` the typed table after the `-- expected` marker.  `imports` is the
    module's own import block, and `body` the statements under the docstring --
    the one spelling both lanes run.
    """

    def __init__(self, path, name, sql, prose, expected, imports, body, skips, xfail):
        self.path = path
        self.name = name
        self.sql = sql
        self.prose = prose
        self.expected = expected
        self.imports = imports
        self.body = body
        self.skips = skips
        self.xfail = xfail

    def __repr__(self):
        return f"Case({self.name!r})"

    @property
    def stem(self):
        return self.name[len(PREFIX) :]

    def skipped(self, lane):
        return lane in self.skips

    def mismatch(self, actual):
        """The expected-vs-actual dump a lane reports when a case disagrees.

        One spelling, because a reader comparing two lanes' failures should not
        have to work out whether the layouts differ for a reason.
        """
        return (
            f"\n--- expected (duckdb) ---\n{self.expected}\n"
            f"--- actual (marrow) ---\n{actual}\n"
        )


class CaseFormat:
    """Reads and writes a case file.

    A case file is a real, standalone Mojo module: the Mojo lane imports it
    rather than copying its body, so what compiles is the file you edit.  That
    costs it a `def` -- Mojo has no top-level statements in a package module --
    but the name is the fixed word `plan`, never the case's own.  **The case's
    identity is its file name**, and nothing inside repeats it.
    """

    MARKER = "-- expected"
    SKIP_MOJO = "-- skip mojo"
    SKIP_PYTHON = "-- skip python"
    XFAIL = "-- xfail "
    DEF = "def plan() raises -> DynRelation:"
    BLANK = "\n\n"

    @classmethod
    def parse(cls, path):
        lines = Path(path).read_text().splitlines()
        try:
            start = lines.index(cls.DEF)
        except ValueError:
            raise SystemExit(f"golden: {path.name}: no `{cls.DEF}` line")

        if len(lines) <= start + 1 or lines[start + 1].strip() != '"""':
            # The opening quotes sit on their own line so the SQL starts at a
            # predictable column and regeneration can rewrite the block without
            # reflowing the first line.
            raise SystemExit(
                f'golden: {path.name}: the line after `{cls.DEF}` must be a lone `"""`'
            )
        for end, line in enumerate(lines[start + 2 :], start + 2):
            if line.strip() == '"""':
                break
        else:
            raise SystemExit(f"golden: {path.name}: docstring is never closed")

        doc = [cls.dedent(line) for line in lines[start + 2 : end]]
        sql, prose, expected, skips, xfail = cls._split_docstring(doc, path.name)
        body = "\n".join(cls.dedent(line) for line in lines[end + 1 :]).strip("\n")
        if not body:
            raise SystemExit(f"golden: {path.name}: case has no body")
        return Case(
            path=path,
            name=PREFIX + path.stem,
            sql=sql,
            prose=prose,
            expected=ExpectedTable.parse(expected, path.name),
            imports="\n".join(lines[:start]).strip("\n"),
            body=body,
            skips=skips,
            xfail=xfail,
        )

    @classmethod
    def _split_docstring(cls, lines, where):
        """`(sql, prose, expected_lines, skips, xfail)` from a docstring.

        `-- xfail <reason>` records a case marrow does not yet answer correctly:
        the query is right and the expectation is DuckDB's, so the corpus states
        the intended behaviour and stays green.  The mark is **strict**, so
        fixing the underlying bug turns the case red and forces the marker's
        removal -- a known bug that quietly starts passing is how a corpus goes
        stale.
        """
        skips = set()
        xfail = None
        kept, expected, in_expected = [], [], False
        for line in lines:
            stripped = line.strip()
            if stripped == cls.MARKER:
                in_expected = True
            elif in_expected:
                expected.append(line)
            elif stripped == cls.SKIP_MOJO:
                skips.add("mojo")
            elif stripped == cls.SKIP_PYTHON:
                skips.add("python")
            elif stripped.startswith(cls.XFAIL):
                xfail = stripped[len(cls.XFAIL) :].strip()
            else:
                kept.append(line)
        if not in_expected:
            raise SystemExit(f"golden: {where}: docstring has no `{cls.MARKER}` block")
        while expected and not expected[-1].strip():
            expected.pop()
        text = "\n".join(kept).strip("\n")
        sql, _, prose = text.partition("\n\n")
        return " ".join(sql.split()), prose.strip("\n"), expected, skips, xfail

    @staticmethod
    def dedent(line):
        """Strip the four spaces of function-body indentation, tabs intact.

        `textwrap.dedent` is not usable here: the expected block is
        tab-separated, and a line whose data begins with a tab would defeat a
        common-prefix calculation.
        """
        if line.startswith("    "):
            return line[4:]
        return line.strip() and line or ""

    @staticmethod
    def indent(text, prefix="    "):
        """Note the asymmetry with `dedent` above: only *that* one needs to be
        hand-written, because the expected block is tab-separated."""
        return textwrap.indent(text, prefix, lambda line: line.strip() != "")

    @classmethod
    def render(cls, case, expected=None):
        """A case back to source, with *expected* replacing its block if given."""
        table = case.expected if expected is None else expected
        doc = [case.sql]
        if case.prose:
            doc.append(case.prose)
        for lane in sorted(case.skips):
            doc.append(f"-- skip {lane}")
        if case.xfail:
            doc.append(f"{cls.XFAIL}{case.xfail}")
        doc.append(f"{cls.MARKER}\n{ExpectedTable.render(table)}")
        return (
            f"{case.imports}\n"
            f"\n\n{cls.DEF}\n"
            f'    """\n'
            f"{cls.indent(cls.BLANK.join(doc))}\n"
            f'    """\n'
            f"{cls.indent(case.body)}\n"
        )


# ---------------------------------------------------------------------------
# Derived artefacts
# ---------------------------------------------------------------------------


class MojoCodegen:
    """Writes the one collectable `test_*.mojo` that runs every case.

    The harness collects by regex-scanning for `def test_*(`, so the wrappers
    have to exist; what they must *not* do is restate the query.  Each is an
    import plus a single call, and the case name -- which `check` needs and Mojo
    cannot introspect -- is supplied here from the file name.
    """

    HEADER = '''"""The golden corpus — one test per case in `{root}/cases/`.

GENERATED by `devkit.golden`; do not edit. Each case is compiled from its own
module and only *called* here, so a case body lives in exactly one place. The
case name comes from the file name.

Regenerate expectations with `pixi run -e bench golden_regenerate`.
"""

from {root}.helpers import check
'''

    def __init__(self, root):
        #: The corpus directory, which is also its Mojo import root -- the
        #: generated module imports the cases and the vocabulary through it.
        self.root = root

    def render(self, cases):
        imports, blocks = [], []
        for case in cases:
            if case.skipped("mojo"):
                continue
            imports.append(
                f"from {self.root}.cases.{case.stem} import plan as _{case.stem}"
            )
            blocks.append(
                f"def {case.name}() raises:\n"
                f'    """{case.sql}"""\n'
                f'    check("{case.name}", _{case.stem}())\n'
            )
        header = self.HEADER.format(root=self.root)
        return header + "\n".join(imports) + "\n\n\n" + "\n\n".join(blocks)

    def write(self, cases, out):
        return write_if_changed(out, self.render(cases))


class PythonTranspiler:
    """The case body as a Python `plan()`.

    Three rules -- drop `var`, drop a transfer sigil in argument position, and
    rewrite a comptime `ddof` parameter to its Python verb -- because the
    signature is written here rather than transcribed, so `raises` and the
    `-> DynRelation` annotation never reach Python and `DynRelation` need not
    exist in its namespace.  The rules hold only while bodies stay inside the
    intersection of the two grammars, which for now means no type annotations.
    """

    VAR = re.compile(r"^(\s*)var\s+")

    DDOF = re.compile(r"\.(stddev|variance)\[([01])\]\(\)")
    """A comptime `ddof` parameter -- `.stddev[1]()` -- as the Python verb.

    `NumericValue.stddev[ddof]` and `.variance[ddof]` take their delta degrees
    of freedom as a *compile-time parameter*, which is the whole reason the
    fused lane can specialise them.  Python has no reading for `[1]` after a
    method, so the two spellings are mapped here: ddof 0 is the population form
    and keeps the base name, ddof 1 is the sample form and takes Arrow's `_samp`
    name -- the same pairing `RuntimeValue` exposes.

    Only 0 and 1 are matched, which is the whole domain marrow implements; a
    case asking for ddof 2 fails loudly rather than transpiling to something
    that silently means ddof 1.
    """

    MOVE = re.compile(r"\b(\w+)\^(?=\s*[,)])")
    """A transfer sigil in argument position -- `join(right^, ...)` -- dropped.

    Narrow on purpose.  It fires only for an *identifier* immediately followed
    by `^` and then a comma or a close paren, which is the one shape Mojo forces
    on a case: `DynRelation.join` takes `var right`, and `DynRelation` is
    deliberately not `ImplicitlyCopyable`, so the sigil is not optional there.

    The two other `^`s in the corpus are untouched by construction.
    `col("p", bool_) ^ col("q", bool_)` is the XOR *operator* -- spaced, and
    preceded by `)` rather than a word character.  `regexp_matches("^[a-z],")`
    has its `^` inside a string, preceded by a quote.  Both fail the lookbehind.

    This rule was missing once, and Python read `right^` as an XOR with a
    missing operand -- a `SyntaxError` that aborts collection for the *whole*
    corpus, not just the five join cases carrying it.
    """

    @staticmethod
    def _ddof(match):
        """`.stddev[1]()` -> `.stddev_samp()`; `.variance[0]()` -> `.variance()`."""
        verb, ddof = match.group(1), match.group(2)
        if ddof == "0":
            return f".{verb}()"
        return ".stddev_samp()" if verb == "stddev" else ".var_samp()"

    @classmethod
    def transpile(cls, case):
        lines = [cls.VAR.sub(r"\1", line) for line in case.body.split("\n")]
        lines = [cls.MOVE.sub(r"\1", line) for line in lines]
        lines = [cls.DDOF.sub(cls._ddof, line) for line in lines]
        doc = case.sql if not case.prose else f"{case.sql}\n\n{case.prose}"
        body = "\n".join(lines)
        return (
            f"def plan():\n"
            f'    """\n'
            f"{CaseFormat.indent(doc)}\n"
            f'    """\n'
            f"{CaseFormat.indent(body)}\n"
        )


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------


class SqlLane:
    """The corpus driven through marrow's SQL front end.

    Every case already carries its query *as SQL* and DuckDB's answer under
    `-- expected`; the other two lanes reach that answer by hand-translating
    the SQL into marrow's verbs.  This one hands the recorded SQL straight to
    `marrow.sql`, which makes the corpus an acceptance suite for the parser at
    no authoring cost -- and makes the hand-translation checkable rather than
    merely suspected.

    **`marrow` is imported inside the methods, never at module scope.**  This
    module is imported by `golden/conftest.py` at collection time, and a
    module-scope import would put a `libmarrow.so` build in front of every
    session that touches the corpus.
    """

    #: A query the front end recognises and refuses carries this marker.  The
    #: prototype covers a subset of SQL, so a refusal is a gap, not a defect --
    #: but anything *without* the marker escaped from inside, and calling those
    #: refusals is exactly how five real bugs once hid in the declined column.
    REFUSAL = "sql:"

    def __init__(self, corpus):
        self.corpus = corpus

    def tables(self):
        """Every fixture as a `marrow.RecordBatch`, keyed by table name.

        Read from the same `.arrow` files the other lanes read, so a divergence
        can never be the fixture.
        """
        import marrow

        return {
            name: marrow.read_ipc_file(str(self.corpus.fixtures.path(name)))[0]
            for name in TABLES
        }

    def run(self, sql):
        """*sql* as a pyarrow table, via the SQL front end."""
        import marrow

        plan = marrow.sql(sql, self.tables())
        return pa.table(plan.to_pyarrow(num_threads=self.corpus.num_threads))

    def outcome(self, case):
        """`(status, detail)` for one case.

        Four outcomes, and the fourth is the one worth explaining.  A case the
        corpus skips in **both** other lanes is one it has already judged
        unassertable -- `temporal_epoch_seconds` is skipped because DuckDB's
        `epoch` returns a DOUBLE that `CAST(... AS BIGINT)` *rounds* while
        marrow truncates, so the expectation encodes the twin's rounding rather
        than anything marrow gets wrong.  Reporting that as a parser failure
        would be reporting someone else's documented decision.  It is
        "divergent".

        A case skipped in both lanes that this one nonetheless *answers* still
        counts as a pass: that is a real capability the others lack.
        """
        try:
            actual = self.run(case.sql)
        except Exception as error:  # noqa: BLE001 - the message is the result
            message = str(error)
            if self.REFUSAL in message:
                return "declined", message[message.index(self.REFUSAL) :].strip()
            return "wrong", f"{type(error).__name__}: {message}".strip()
        if actual.equals(case.expected):
            return "pass", ""
        if case.skipped("mojo") and case.skipped("python"):
            return "divergent", "the corpus skips this case in both lanes"
        if actual.column_names != case.expected.column_names:
            return (
                "wrong",
                f"columns {actual.column_names} != {case.expected.column_names}",
            )
        return "wrong", case.mismatch(actual)

    def coverage(self):
        """`(counts, reasons, wrong)` over the whole corpus."""
        counts = Counter()
        reasons = Counter()
        wrong = []
        for case in self.corpus.cases():
            status, detail = self.outcome(case)
            counts[status] += 1
            if status == "declined":
                # The first line is the message; group by it so the table shows
                # *kinds* of gap rather than one sentence per case.
                reasons[detail.split("\n")[0][:72]] += 1
            elif status == "wrong":
                wrong.append(case.stem)
        return counts, reasons, wrong

    def report(self, echo=print):
        """The coverage table, returning the cases that answered wrongly.

        The returned list is the one the table prints, not a second count of
        it: the caller gates on exactly what a reader sees.  Deriving the two
        separately let a summary claim "0 wrong" while listing every case
        underneath it.
        """
        counts, reasons, wrong = self.coverage()
        total = sum(counts.values())
        echo(f"golden cases: {total}")
        echo(f"  pass:      {counts['pass']:4d}  ({counts['pass'] / total:.0%})")
        echo(f"  declined:  {counts['declined']:4d}")
        echo(f"  divergent: {counts['divergent']:4d}")
        echo(f"  wrong:     {len(wrong):4d}")
        if reasons:
            echo("\ndeclined, by reason:")
            for reason, count in reasons.most_common():
                echo(f"  {count:4d}  {reason}")
        if wrong:
            echo("\nwrong answers:")
            for stem in wrong:
                echo(f"        {stem}")
        return wrong


class Corpus:
    """Everything the golden corpus needs, over one checkout.

    The Mojo lane's wrappers and the expectation files are build output, so they
    live in gitignored subdirectories rather than beside the sources.  A tmpdir
    is not reachable for the wrappers, and the blocker is pytest's rather than
    the compiler's: `pytest_collect_file` applies only to files under the
    repository conftest's own directory, so a `.mojo` in /tmp is never
    recognised as a test file.
    """

    def __init__(self, repo):
        self.repo = repo
        self.directory = repo.golden_dir
        self.fixtures = FixtureSet(self.directory / "fixtures")
        self._cases = None
        #: CPU worker budget for a corpus run, set once from the command line.
        #: It rides here rather than in a second module global because the
        #: vocabulary already reaches the corpus and nothing else wants it.
        self.num_threads = 0

    @property
    def case_dir(self):
        return self.directory / "cases"

    @property
    def expectation_dir(self):
        return self.directory / ".exp"

    @property
    def generated(self):
        return self.directory / "generated" / "test_cases.mojo"

    def cases(self):
        if self._cases is None:
            if not self.case_dir.is_dir():
                raise SystemExit(f"golden: no case directory at {self.case_dir}")
            self._cases = [
                CaseFormat.parse(path) for path in sorted(self.case_dir.glob("*.mojo"))
            ]
        return self._cases

    def xfail_reasons(self):
        return {case.name: case.xfail for case in self.cases() if case.xfail}

    def prepare(self):
        """Everything collection depends on.  Called from `conftest.py` on import."""
        cases = self.cases()
        self.fixtures.write()
        self.write_expectations()
        MojoCodegen(self.directory.name).write(cases, self.generated)
        return cases

    def write_expectations(self):
        """One Arrow IPC file per case -- what the Mojo lane compares against.

        Derived from the committed case text, so it cannot drift from what a
        reviewer saw.  Mojo has no JSON library and a typed IPC file needs no
        parser.
        """
        self.expectation_dir.mkdir(parents=True, exist_ok=True)
        for case in self.cases():
            table = case.expected
            # `write_table` emits *no* batch for a zero-row table, and an empty
            # result is a normal query outcome -- so write batches explicitly
            # and synthesise one empty batch when there are none. Otherwise the
            # Mojo lane reads a file with 0 batches and cannot tell "empty
            # result" from "expectation missing".
            batches = table.to_batches() or [
                pa.record_batch(
                    [pa.array([], type=field.type) for field in table.schema],
                    schema=table.schema,
                )
            ]
            path = self.expectation_dir / f"{case.name}.arrow"
            with pa.ipc.new_file(path, table.schema) as writer:
                for batch in batches:
                    writer.write_batch(batch)

    def install_python_lane(self, namespace, check, target):
        """Compile every case against *namespace* and inject a test into *target*.

        Injecting real functions rather than parametrising one test keeps the
        item names identical to the Mojo lane's, so `-k` selects the same cases
        in both.  `check` is passed in rather than imported: it lives in
        `golden/helpers.py`, which imports this module, and closing that cycle
        would be an import-order trap.
        """
        import pytest

        for case in self.cases():
            scope = dict(namespace)
            exec(
                compile(PythonTranspiler.transpile(case), str(case.path), "exec"), scope
            )
            function = self._wrap(case, scope["plan"], check)
            if case.skipped("python"):
                function = pytest.mark.skip(reason="-- skip python")(function)
            target[case.name] = function

    @staticmethod
    def _wrap(case, plan, check):
        """A pytest function for one case.  Its own scope, so the loop cannot leak."""

        def run():
            check(case.name, plan())

        run.__name__ = case.name
        run.__qualname__ = case.name
        run.__doc__ = case.sql
        return run

    def regenerate(self, echo=print):
        """Rewrite every case's expected block from DuckDB.

        Expectations come from **DuckDB**, never from marrow: an expectation
        captured from the engine under test enshrines whatever that engine
        currently does, which is how a golden corpus quietly becomes a record of
        its own bugs.
        """
        import duckdb  # noqa: PLC0415 -- see the module docstring

        written = self.fixtures.write()
        echo(f"fixtures: {', '.join(written)}")

        connection = duckdb.connect()
        for name in written:
            connection.register(name, self.fixtures.read(name))

        for case in self.cases():
            table = pa.table(connection.execute(case.sql).arrow())
            case.path.write_text(CaseFormat.render(case, table))
            echo(f"  {case.stem}: {table.num_rows} rows")
