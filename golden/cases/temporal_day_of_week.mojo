from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(isodow(ts) - 1 AS INTEGER) AS dow FROM events

    A recorded divergence, and the twin is written to ask DuckDB *marrow's*
    question.

    Three conventions are in play. marrow returns the ISO weekday with
    **Monday = 0** — `DayOfWeekKernel.component` in `marrow/kernels/temporal.mojo`
    comments "ISO weekday, Monday = 0 (PyArrow default)" — matching
    `pyarrow.compute.day_of_week`'s defaults (`count_from_zero=True`,
    `week_start=1`). DuckDB's `dayofweek` is **Sunday = 0**, and its `isodow` is
    **Monday = 1**. So `isodow(ts) - 1` is DuckDB's spelling of marrow's answer.

    Writing the twin as a bare `dayofweek(ts)` would assert DuckDB's Sunday-first
    convention and report Arrow-correct behaviour as a defect. Note that this
    fixture cannot tell the two apart on its own: its dates are Friday, Tuesday
    and Saturday, and `dayofweek` and `isodow` happen to agree on all three —
    they differ only on a Sunday. The convention was read off the kernel, not
    inferred from the rows.

    -- expected
    dow:int32
    4
    1
    1
    NULL
    5
    4
    """
    var t = table("events")
    return t.project(["dow"], [col("ts", timestamp(microsecond)).day_of_week()])
