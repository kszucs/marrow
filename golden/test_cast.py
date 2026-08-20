"""Golden cases — casts, the runtime lane.

Where DuckDB and Arrow are most likely to disagree on purpose: rounding
versus truncation on float→int, what an unparseable string becomes, and
whether a conversion carries validity through.

Every case uses ``safe=False``. With the default ``safe=True`` a lossy
conversion raises, which is a different question — worth its own cases, but
not comparable against a SQL CAST that simply converts.
"""

import marrow as ma


def test_golden_cast_int_to_float(golden):
    """SELECT CAST(i AS DOUBLE) AS c FROM nums"""
    t = golden.table("nums")
    golden.check(t.project(c=t["i"].cast(ma.float64(), safe=False)))


def test_golden_cast_int_to_int32(golden):
    """SELECT CAST(i AS INTEGER) AS c FROM nums"""
    t = golden.table("nums")
    golden.check(t.project(c=t["i"].cast(ma.int32(), safe=False)))


def test_golden_cast_float_to_int(golden):
    """SELECT CAST(TRUNC(f) AS BIGINT) AS c FROM nums

    A recorded divergence. DuckDB's `CAST(1.7 AS BIGINT)` **rounds** to 2 and
    `0.5` to 0 (half-to-even); Arrow **truncates** toward zero, giving 1 and
    0. PyArrow confirms marrow's answer — `pc.cast(..., safe=False)` returns
    `[1, -2, 0, None]` and `safe=True` raises "Float value 1.700000 was
    truncated converting to int64".

    So the twin says `TRUNC` to ask DuckDB the question marrow answers.
    Writing the twin as a bare CAST would assert DuckDB's rounding rule and
    report Arrow-correct behaviour as a defect.
    """
    t = golden.table("nums")
    golden.check(t.project(c=t["f"].cast(ma.int64(), safe=False)))


def test_golden_cast_int_to_string(golden):
    """SELECT CAST(i AS VARCHAR) AS c FROM nums"""
    t = golden.table("nums")
    golden.check(t.project(c=t["i"].cast(ma.string(), safe=False)))


def test_golden_cast_string_to_int(golden):
    """SELECT TRY_CAST(s AS BIGINT) AS c FROM nums

    `TRY_CAST`, because 'abc' does not parse: DuckDB's plain CAST raises and
    marrow nulls the value, so the twin has to ask DuckDB the same question
    marrow answers.
    """
    t = golden.table("nums")
    golden.check(t.project(c=t["s"].cast(ma.int64(), safe=False)))


def test_golden_cast_bool_to_int(golden):
    """SELECT CAST(b AS BIGINT) AS c FROM nums"""
    t = golden.table("nums")
    golden.check(t.project(c=t["b"].cast(ma.int64(), safe=False)))


def test_golden_cast_int_to_bool(golden):
    """SELECT CAST(i AS BOOLEAN) AS c FROM nums"""
    t = golden.table("nums")
    golden.check(t.project(c=t["i"].cast(ma.bool_(), safe=False)))
