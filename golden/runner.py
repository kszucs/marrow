"""The golden corpus: fixtures, case format, codegen, transpile, regeneration.

One case is one file, `golden/cases/<name>.mojo`, and it is Mojo source. The
Mojo lane compiles it ahead of time; the Python lane runs the *same text*
through a mechanical transpile and the runtime bindings. Keeping one spelling
is the point — the two lanes drifted once already, when the AOT lane had no
boolean column leaf and the Python twin silently tested something else.

This module owns everything that is not the case vocabulary itself (that is
`helpers.py` for Python and `helpers.mojo` for Mojo):

* the fixture tables and their on-disk Arrow files,
* the case-file format — parse and render,
* the derived artefacts, `.exp/<case>.arrow` and `test_cases.mojo`,
* the Mojo -> Python transpile,
* regeneration of expectations from DuckDB.

**`duckdb` is imported inside `regenerate()`, never at module scope.** The
`dev` environment has no duckdb and comparing against an expectation does not
need one; only producing an expectation does. An earlier layout split
`expfmt.py` out of `regenerate.py` for exactly this reason, and folding them
back together is only safe while that import stays where it is.
"""

import ast
import re
import sys
from pathlib import Path

import pyarrow as pa

DIR = Path(__file__).parent
CASES = DIR / "cases"
FIXTURES = DIR / "fixtures"
CACHE = DIR / ".exp"
GENERATED = DIR / "test_cases.mojo"

PREFIX = "test_golden_"
MARKER = "-- expected"
SKIP_MOJO = "-- skip mojo"
SKIP_PYTHON = "-- skip python"

# Set from `conftest.py`'s `pytest_configure`; the `Golden` fixture that used
# to read them off `request.config` is gone, so that `table(name)` reads the
# same in both lanes instead of carrying a config argument into every case.
MORSEL_SIZE = 8192
NUM_THREADS = 0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
# Fixtures are **files**, not construction code: all three consumers — the AOT
# lane, the runtime lane, and the DuckDB twin producing the expectations — read
# the same bytes. Building the table per lane would let them drift on exactly
# the thing under test.
#
# Nulls appear in every column that can hold one. A fixture without them tests
# the happy path of kernels whose null handling is the interesting part.

TABLES = {
    # `k` repeats so grouping has something to do, and carries a null key — the
    # case Arrow and SQL engines disagree about most often. `v` and `w` carry
    # nulls at different rows so a binary op sees each side null independently.
    "basic": pa.table(
        {
            "k": pa.array(["a", "b", "a", "c", "b", "a", None], pa.string()),
            "v": pa.array([1, 2, 3, 4, None, 6, 7], pa.int64()),
            "w": pa.array([10, None, 30, 40, 50, 60, 70], pa.int64()),
        }
    ),
    # Null semantics. `a` is entirely null, `b` has none, so an aggregate over
    # `a` exercises "no valid input" and one over `b` gives an exact mean
    # (20 / 4 = 5.0) — a non-exact one would compare two double roundings
    # rather than two implementations.
    "nulls": pa.table(
        {
            "a": pa.array([None, None, None, None], pa.int64()),
            "b": pa.array([2, 4, 6, 8], pa.int64()),
            "g": pa.array(["x", None, "x", "y"], pa.string()),
        }
    ),
    # Three-valued logic through *derived* predicates (`x > 0`, `y > 0`),
    # covering the full 3x3 Kleene table.
    #
    #   x > 0 : T T T F F F N N N
    #   y > 0 : T F N T F N T F N
    "kleene": pa.table(
        {
            "x": pa.array([1, 1, 1, -1, -1, -1, None, None, None], pa.int64()),
            "y": pa.array([1, -1, None, 1, -1, None, 1, -1, None], pa.int64()),
        }
    ),
    # Join fixtures. `emp.dept` covers every interesting case against
    # `dept.did`: a unique match (10), a duplicated match (20, twice), a key
    # with no match (99), and a NULL key — which must match nothing, not even
    # another NULL. `dept.did` 30 is unmatched from the right, so outer joins
    # have something to widen in both directions.
    "emp": pa.table(
        {
            "eid": pa.array([1, 2, 3, 4, 5], pa.int64()),
            "dept": pa.array([10, 20, 20, 99, None], pa.int64()),
        }
    ),
    "dept": pa.table(
        {
            "did": pa.array([10, 20, 30], pa.int64()),
            "dname": pa.array(["eng", "sales", "ops"], pa.string()),
        }
    ),
    # Strings worth asking questions about: mixed case, surrounding
    # whitespace, the empty string (which is not a null), a multi-byte
    # character so `length` has to say whether it counts bytes or codepoints,
    # and a null.
    "words": pa.table(
        {"s": pa.array(["Hello", "wORLD", "  pad  ", "", "héllo", None], pa.string())}
    ),
    # Actual boolean columns, covering the 3x3 Kleene table directly rather
    # than through derived predicates.
    "flags": pa.table(
        {
            "p": pa.array(
                [True, True, True, False, False, False, None, None, None],
                pa.bool_(),
            ),
            "q": pa.array(
                [True, False, None, True, False, None, True, False, None],
                pa.bool_(),
            ),
        }
    ),
    # Cast inputs. `f` holds values where truncation and rounding disagree
    # (1.7, -2.7, 0.5); `s` holds one string that does not parse; `i` holds a
    # value too wide for int32 is deliberately absent — 300 fits, so the
    # int32 case tests conversion rather than overflow.
    "nums": pa.table(
        {
            "i": pa.array([1, -2, 300, None], pa.int64()),
            "f": pa.array([1.7, -2.7, 0.5, None], pa.float64()),
            "s": pa.array(["1", "-2", "abc", None], pa.string()),
            "b": pa.array([True, False, True, None], pa.bool_()),
        }
    ),
}


def fixture_path(name):
    return FIXTURES / f"{name}.arrow"


def write_fixtures():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for name, table in TABLES.items():
        with pa.ipc.new_file(fixture_path(name), table.schema) as writer:
            writer.write_table(table)
    return sorted(TABLES)


def read_fixture(name):
    with pa.ipc.open_file(fixture_path(name)) as reader:
        return reader.read_all()


# ---------------------------------------------------------------------------
# The typed-TSV expectation block
# ---------------------------------------------------------------------------
# A type per column is what lets the text be read back into a *typed* Arrow
# table. sqllogictest's `query IIR` letters coerce results before comparing,
# which hides exactly the type bugs an Arrow engine should be asserting.

NULL = "NULL"

# Only the types the corpus uses are mapped; an unmapped type is an error
# rather than a silent cast.
TYPES = {
    "string": pa.string(),
    "int64": pa.int64(),
    "int32": pa.int32(),
    "double": pa.float64(),
    "bool": pa.bool_(),
}
_NAMES = {str(v): k for k, v in TYPES.items()}


def type_name(dtype):
    try:
        return _NAMES[str(dtype)]
    except KeyError:
        raise SystemExit(f"golden: unmapped arrow type {dtype!r}; add it to TYPES")


def render_value(value):
    """One cell. Strings are **quoted**, and that is load-bearing.

    `mojo format` strips trailing whitespace inside a docstring. The `words`
    fixture holds `"  pad  "`, so an unquoted block lost the trailing spaces
    and the expectation silently became a different string — a corpus that
    asserts the wrong answer. Quoting keeps every line ending in a printable
    character. It also lets string data contain a tab or a newline, which the
    bare format could not represent at all.
    """
    if value is None:
        return NULL
    if isinstance(value, (str, float)):
        return repr(value)
    return str(value)


def parse_value(text, dtype):
    if text == NULL:
        return None
    if dtype == pa.bool_():
        return text == "True"
    return ast.literal_eval(text)


def render_expected(table):
    """A pyarrow table -> the tab-separated block that goes in a docstring."""
    header = "\t".join(f"{f.name}:{type_name(f.type)}" for f in table.schema)
    rows = [
        "\t".join(render_value(v) for v in row.values()) for row in table.to_pylist()
    ]
    return "\n".join([header, *rows])


def parse_expected(lines, where):
    """The block back into a typed pyarrow table."""
    if not lines:
        raise SystemExit(f"golden: {where}: empty `{MARKER}` block")
    columns, types = [], []
    for spec in lines[0].split("\t"):
        column, _, tname = spec.partition(":")
        if tname not in TYPES:
            raise SystemExit(f"golden: {where}: unknown type {tname!r} in header")
        columns.append(column)
        types.append(TYPES[tname])
    rows = [line.split("\t") for line in lines[1:]]
    for row in rows:
        if len(row) != len(columns):
            raise SystemExit(
                f"golden: {where}: row has {len(row)} fields, header has {len(columns)}"
            )
    return pa.table(
        {
            column: pa.array([parse_value(row[i], types[i]) for row in rows], types[i])
            for i, column in enumerate(columns)
        }
    )


# ---------------------------------------------------------------------------
# The case-file format
# ---------------------------------------------------------------------------

# A case file is a real, standalone Mojo module: the Mojo lane imports it
# rather than copying its body, so what compiles is the file you edit. That
# costs it a `def` — Mojo has no top-level statements in a package module — but
# the name is the fixed word `plan`, never the case's own. **The case's
# identity is its file name**, and nothing inside repeats it.
_DEF = "def plan() raises -> DynRelation:"
_BLANK = "\n\n"


class Case:
    """One `golden/cases/<stem>.mojo`.

    `sql` is the docstring's first paragraph, `prose` whatever follows it, and
    `expected` the typed table after the `-- expected` marker. `imports` is the
    module's own import block, and `body` the statements under the docstring —
    the one spelling both lanes run.
    """

    def __init__(self, path, name, sql, prose, expected, imports, body, skips):
        self.path = path
        self.name = name
        self.sql = sql
        self.prose = prose
        self.expected = expected
        self.imports = imports
        self.body = body
        self.skips = skips

    @property
    def stem(self):
        return self.name[len(PREFIX) :]

    def skipped(self, lane):
        return lane in self.skips


def _split_docstring(lines, where):
    """`(sql, prose, expected_lines, skips)` from a case's docstring lines."""
    skips = set()
    kept, expected, in_expected = [], [], False
    for line in lines:
        stripped = line.strip()
        if stripped == MARKER:
            in_expected = True
        elif in_expected:
            expected.append(line)
        elif stripped == SKIP_MOJO:
            skips.add("mojo")
        elif stripped == SKIP_PYTHON:
            skips.add("python")
        else:
            kept.append(line)
    if not in_expected:
        raise SystemExit(f"golden: {where}: docstring has no `{MARKER}` block")
    while expected and not expected[-1].strip():
        expected.pop()
    text = "\n".join(kept).strip("\n")
    sql, _, prose = text.partition("\n\n")
    return " ".join(sql.split()), prose.strip("\n"), expected, skips


def parse_case(path):
    lines = path.read_text().splitlines()
    try:
        start = lines.index(_DEF)
    except ValueError:
        raise SystemExit(f"golden: {path.name}: no `{_DEF}` line")

    if len(lines) <= start + 1 or lines[start + 1].strip() != '"""':
        # The opening quotes sit on their own line so the SQL starts at a
        # predictable column and regeneration can rewrite the block without
        # reflowing the first line.
        raise SystemExit(
            f'golden: {path.name}: the line after `{_DEF}` must be a lone `"""`'
        )
    for end, line in enumerate(lines[start + 2 :], start + 2):
        if line.strip() == '"""':
            break
    else:
        raise SystemExit(f"golden: {path.name}: docstring is never closed")

    doc = [_dedent(line) for line in lines[start + 2 : end]]
    sql, prose, expected_lines, skips = _split_docstring(doc, path.name)
    body = "\n".join(_dedent(line) for line in lines[end + 1 :]).strip("\n")
    if not body:
        raise SystemExit(f"golden: {path.name}: case has no body")
    return Case(
        path=path,
        name=PREFIX + path.stem,
        sql=sql,
        prose=prose,
        expected=parse_expected(expected_lines, path.name),
        imports="\n".join(lines[:start]).strip("\n"),
        body=body,
        skips=skips,
    )


def _dedent(line):
    """Strip the four spaces of function-body indentation, tabs intact.

    `textwrap.dedent` is not usable here: the expected block is tab-separated,
    and a line whose data begins with a tab would defeat a common-prefix
    calculation.
    """
    if line.startswith("    "):
        return line[4:]
    return line.strip() and line or ""


def _indent(text, prefix="    "):
    return "\n".join(prefix + line if line else "" for line in text.split("\n"))


def render_case(case, expected=None):
    """A case back to source, with `expected` replacing its block if given."""
    table = case.expected if expected is None else expected
    doc = [case.sql]
    if case.prose:
        doc.append(case.prose)
    for lane in sorted(case.skips):
        doc.append(f"-- skip {lane}")
    doc.append(f"{MARKER}\n{render_expected(table)}")
    return (
        f"{case.imports}\n"
        f"\n\n{_DEF}\n"
        f'    """\n'
        f"{_indent(_BLANK.join(doc))}\n"
        f'    """\n'
        f"{_indent(case.body)}\n"
    )


def load_cases():
    if not CASES.is_dir():
        raise SystemExit(f"golden: no case directory at {CASES}")
    return [parse_case(p) for p in sorted(CASES.glob("*.mojo"))]


# ---------------------------------------------------------------------------
# Derived artefacts
# ---------------------------------------------------------------------------

# Every name a case body may use, imported once for the whole generated
# module. Case files carry no imports of their own: this list and
# `helpers.NAMESPACE` are the same vocabulary written down per lane, which is
# what makes "the two lanes run one spelling" checkable rather than hopeful.
GENERATED_HEADER = '''"""The golden corpus — one test per case in `golden/cases/`.

GENERATED by `golden/runner.py`; do not edit. Each case is compiled from its
own module and only *called* here, so a case body lives in exactly one place.
The case name comes from the file name.

Regenerate expectations with `pixi run -e bench python golden/runner.py`.
"""

from golden.helpers import check
'''


def generate_mojo(cases):
    """One collectable `test_*.mojo` that imports every case and checks it.

    The harness collects by regex-scanning for `def test_*(`, so the wrappers
    have to exist; what they must *not* do is restate the query. Each is an
    import plus a single call, and the case name — which `check` needs and Mojo
    cannot introspect — is supplied here from the file name.
    """
    imports, blocks = [], []
    for case in cases:
        if case.skipped("mojo"):
            continue
        imports.append(f"from golden.cases.{case.stem} import plan as _{case.stem}")
        blocks.append(
            f"def {case.name}() raises:\n"
            f'    """{case.sql}"""\n'
            f'    check("{case.name}", _{case.stem}())\n'
        )
    text = GENERATED_HEADER + "\n".join(imports) + "\n\n\n" + "\n\n".join(blocks)
    # Leave an unchanged file alone so the harness's content-addressed driver
    # cache is not invalidated on every run.
    if not GENERATED.exists() or GENERATED.read_text() != text:
        GENERATED.write_text(text)


def write_expectations(cases):
    """One Arrow IPC file per case — what the Mojo lane compares against.

    Derived from the committed case text, so it cannot drift from what a
    reviewer saw. Mojo has no JSON library and a typed IPC file needs no
    parser.
    """
    CACHE.mkdir(exist_ok=True)
    for case in cases:
        table = case.expected
        # `write_table` emits *no* batch for a zero-row table, and an empty
        # result is a normal query outcome — so write batches explicitly and
        # synthesise one empty batch when there are none. Otherwise the Mojo
        # lane reads a file with 0 batches and cannot tell "empty result" from
        # "expectation missing".
        batches = table.to_batches() or [
            pa.record_batch(
                [pa.array([], type=f.type) for f in table.schema],
                schema=table.schema,
            )
        ]
        with pa.ipc.new_file(CACHE / f"{case.name}.arrow", table.schema) as writer:
            for batch in batches:
                writer.write_batch(batch)


def prepare():
    """Everything collection depends on. Called from `conftest.py` on import."""
    cases = load_cases()
    write_fixtures()
    write_expectations(cases)
    generate_mojo(cases)
    return cases


# ---------------------------------------------------------------------------
# Mojo -> Python
# ---------------------------------------------------------------------------

_VAR_RE = re.compile(r"^(\s*)var\s+")


def transpile(case):
    """The case body as a Python `plan()`.

    One rule — drop `var` — because the signature is written here rather than
    transcribed, so `raises` and the `-> DynRelation` annotation never reach
    Python and `DynRelation` need not exist in its namespace. The rule holds
    only while bodies stay inside the intersection of the two grammars: no type
    annotations, and no transfer sigils (`q^` has no Python reading, so every
    helper a case calls takes its arguments borrowed).
    """
    lines = [_VAR_RE.sub(r"\1", line) for line in case.body.split("\n")]
    doc = case.sql if not case.prose else f"{case.sql}\n\n{case.prose}"
    return (
        f"def plan():\n"
        f'    """\n'
        f"{_indent(doc)}\n"
        f'    """\n'
        f"{_indent(chr(10).join(lines))}\n"
    )


def install(namespace, check, target, cases=None):
    """Compile every case against `namespace` and inject a test into `target`.

    Injecting real functions rather than parametrising one test keeps the item
    names identical to the Mojo lane's, so `-k` selects the same cases in both.
    `check` is passed in rather than imported: it lives in `helpers`, which
    imports this module, and closing that cycle would be a import-order trap.
    """
    import pytest

    for case in load_cases() if cases is None else cases:
        scope = dict(namespace)
        exec(compile(transpile(case), str(case.path), "exec"), scope)
        function = _wrap(case, scope["plan"], check)
        if case.skipped("python"):
            function = pytest.mark.skip(reason="-- skip python")(function)
        target[case.name] = function


def _wrap(case, plan, check):
    """A pytest function for one case. Its own scope, so the loop cannot leak."""

    def run():
        check(case.name, plan())

    run.__name__ = case.name
    run.__qualname__ = case.name
    run.__doc__ = case.sql
    return run


# ---------------------------------------------------------------------------
# Regeneration
# ---------------------------------------------------------------------------


def regenerate():
    """Rewrite every case's expected block from DuckDB.

    Expectations come from **DuckDB**, never from marrow: an expectation
    captured from the engine under test enshrines whatever that engine
    currently does, which is how a golden corpus quietly becomes a record of
    its own bugs.
    """
    import duckdb  # noqa: PLC0415 — see the module docstring

    written = write_fixtures()
    print(f"fixtures: {', '.join(written)}")

    connection = duckdb.connect()
    for name in written:
        connection.register(name, read_fixture(name))

    for case in load_cases():
        table = pa.table(connection.execute(case.sql).arrow())
        case.path.write_text(render_case(case, table))
        print(f"  {case.stem}: {table.num_rows} rows")


if __name__ == "__main__":
    sys.exit(regenerate())
