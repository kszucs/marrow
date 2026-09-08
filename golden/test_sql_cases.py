"""The golden corpus, driven through the SQL front end.

Every case in `golden/cases/` already carries the query **as SQL** in its
docstring and DuckDB's answer under `-- expected`; the two other lanes reach
that answer by hand-translating the SQL into marrow's verbs. This lane skips
the translation and hands the recorded SQL to `marrow.sql`, which makes the
corpus an acceptance suite for the parser at no authoring cost — and makes the
hand-translation, which `backlog.md` calls "evidence of the impedance",
checkable rather than merely suspected.

The lane itself is `devkit.golden.SqlLane`; this file is only its pytest
surface. The coverage table — what the parser covers, and what each declined
case tripped over — is the list to work down when widening the grammar:

    pixi run -e dev pytest golden/test_sql_cases.py -q   # the suite
    pixi run -e dev golden_sql                           # the coverage table

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
"""

import pytest

from devkit.golden import SqlLane, corpus

_LANE = SqlLane(corpus())
_CASES = corpus().cases()


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.stem)
def test_sql(case):
    """The recorded SQL must reach the recorded answer, or say why it cannot."""
    status, detail = _LANE.outcome(case)
    if status in ("declined", "divergent"):
        pytest.skip(detail)
    assert status == "pass", detail
