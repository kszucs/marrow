"""The golden corpus, driven through the SQL front end.

Every case in `golden/cases/` already carries the query **as SQL** in its
docstring and DuckDB's answer under `-- expected`; the two existing lanes reach
that answer by hand-translating the SQL into marrow's verbs. This lane skips
the translation and hands the recorded SQL to `marrow.sql`, which makes the
corpus an acceptance suite for the parser at no authoring cost — and makes the
hand-translation, which `backlog.md` calls "evidence of the impedance",
checkable rather than merely suspected.

Named `test_sql_cases` rather than `test_sql` because `golden/` is not a
package: pytest resolves a test module by basename, and a second `test_sql.py`
anywhere in the tree — `python/marrow/tests/` has one — collides at collection
and interrupts the whole run.

**A query the front end declines is a skip, not a failure.** The prototype
covers a subset of SQL, and a case using something outside it raises an error
carrying `sql:` — a refusal, which this file reports as a skip naming the
reason. Anything else is a real failure, whether it answered differently from
DuckDB or escaped with an internal error: a planner defect surfaces as a
message with no `sql:` in it, and treating those as skips is exactly how one
would hide.

    pixi run -e dev pytest golden/test_sql_cases.py -q    # the suite
    pixi run -e dev python golden/test_sql_cases.py       # the coverage table

The second form prints what is covered and what each declined case tripped
over, which is the list to work down when widening the grammar.
"""

import collections

import pyarrow as pa
import pytest

import marrow
import runner


def _tables():
    """Every fixture as a `marrow.RecordBatch`, keyed by table name.

    Read from the same `.arrow` files the other lanes read, so a divergence
    can never be the fixture.
    """
    return {
        name: marrow.read_ipc_file(str(runner.fixture_path(name)))[0]
        for name in runner.TABLES
    }


def _run(sql):
    """`sql` as a pyarrow table, via the SQL front end."""
    return pa.table(
        marrow.sql(sql, _tables()).to_pyarrow(num_threads=runner.NUM_THREADS)
    )


def _outcome(case):
    """`(status, detail)` for one case.

    Four outcomes, and the fourth is the one worth explaining. A case the
    corpus skips in **both** existing lanes is one it has already judged
    unassertable — `temporal_epoch_seconds` is skipped because DuckDB's
    `epoch` returns a DOUBLE that `CAST(... AS BIGINT)` *rounds* while marrow
    truncates, so the expectation encodes the twin's rounding rather than
    anything marrow gets wrong. Reporting that as a parser failure would be
    reporting someone else's documented decision. It is "divergent".

    A case skipped in both lanes that this one nonetheless *answers* still
    counts as a pass, because that is a real capability the other lanes lack.
    """
    try:
        actual = _run(case.sql)
    except Exception as error:  # noqa: BLE001 - the message is the result
        message = str(error)
        if "sql:" in message:
            # A refusal: the front end recognised the query and said no.
            return "declined", message[message.index("sql:") :].strip()
        # Anything else escaped from inside — `column '' not found in schema`
        # is a planner defect, not an unsupported grammar — and reporting it
        # as a skip is how five real bugs hid in the "declined" column.
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
    return "wrong", f"\n--- expected ---\n{case.expected}\n--- actual ---\n{actual}"


_CASES = runner.load_cases()


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.stem)
def test_sql(case):
    """The recorded SQL must reach the recorded answer, or say why it cannot."""
    status, detail = _outcome(case)
    if status in ("declined", "divergent"):
        pytest.skip(detail)
    assert status == "pass", detail


def main():
    """Print the coverage table. Not a test — the suite above is."""
    counts = collections.Counter()
    reasons = collections.Counter()
    wrong = []
    for case in _CASES:
        status, detail = _outcome(case)
        counts[status] += 1
        if status == "declined":
            # The first line is the message; group by it so the table shows
            # *kinds* of gap rather than 200 individual sentences.
            reasons[detail.split("\n")[0][:72]] += 1
        elif status == "wrong":
            wrong.append(case.stem)

    total = sum(counts.values())
    print(f"golden cases: {total}")
    print(f"  pass:      {counts['pass']:4d}  ({counts['pass'] / total:.0%})")
    print(f"  declined:  {counts['declined']:4d}")
    print(f"  divergent: {counts['divergent']:4d}")
    print(f"  wrong:     {counts['wrong']:4d}")
    if reasons:
        print("\ndeclined, by reason:")
        for reason, n in reasons.most_common():
            print(f"  {n:4d}  {reason}")
    if wrong:
        print("\nwrong answers:")
        for stem in wrong:
            print(f"        {stem}")


if __name__ == "__main__":
    main()
