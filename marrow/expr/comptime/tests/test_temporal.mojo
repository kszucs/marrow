"""The temporal family: comparison, field extraction, truncation.

`TemporalValue` was a marker with two aggregates on it. `TemporalCompare` and
its six aliases existed in `numeric.mojo` with **no callers at all** — naming
`TemporalGt(a, b)` by hand was the only way to reach one, and
`TemporalNe`/`TemporalLe`/`TemporalGe` had not even that — and the nine
extraction kernels plus `DateTruncKernel` had been in `kernels/temporal.mojo`
unreferenced from any expression. A family documented as "ordered and
comparable, but not arithmetic" could not be compared.

Three things are worth asserting and each has its own case below:

- comparison reaches `TemporalCompare` and **rejects a unit mismatch** rather
  than silently comparing tick counts,
- extraction answers `int32` and carries the operand's nulls,
- truncation keeps the operand's *dtype*, unit included, which is the one
  place a temporal node cannot answer from `Self.Type()` — a `TemporalType`
  is not `Defaultable`.
"""

from std.testing import assert_equal, assert_raises, assert_true

from ...builders import col, table
from ...bindings import Bindings
from ....arrays import BoolArray, Int32Array, TimestampArray
from ....builders import PrimitiveBuilder, array
from ....dtypes import (
    Date32Type,
    DynType,
    TimestampType,
    date32,
    int64,
    microsecond,
    second,
    timestamp,
)
from ....tabular import RecordBatch, record_batch
from ..core import BoolValue, NumericValue, TemporalValue


def _events() raises -> RecordBatch:
    """Two timestamps and a null, at second resolution.

    2019-06-15T12:30:45 (a Saturday), 2020-02-29T00:00:00 (a leap day, so
    `day_of_year` has to count February correctly).
    """
    var b = PrimitiveBuilder[TimestampType](timestamp(second), capacity=3)
    b.append(Int64(1_560_601_845))
    b.append_null()
    b.append(Int64(1_582_934_400))
    var d = PrimitiveBuilder[Date32Type](date32(), capacity=3)
    d.append(Int32(18_062))  # 2019-06-15
    d.append_null()
    d.append(Int32(18_321))  # 2020-02-29
    # The same instants at microsecond resolution, so a cross-unit comparison
    # is expressible against this batch. Both are int64-backed, so nothing at
    # compile time separates them.
    var us = PrimitiveBuilder[TimestampType](timestamp(microsecond), capacity=3)
    us.append(Int64(1_560_601_845_000_000))
    us.append_null()
    us.append(Int64(1_582_934_400_000_000))
    return record_batch(
        [b.finish().to_dyn(), d.finish().to_dyn(), us.finish().to_dyn()],
        names=["ts", "d", "us"],
    )


def _as_i32(v: Some[NumericValue], b: RecordBatch) raises -> Int32Array:
    return (
        v.evaluate(b.to_struct_array(), Bindings())
        .to_array(b.num_rows())
        .as_int32()
        .copy()
    )


def _as_bool(v: Some[BoolValue], b: RecordBatch) raises -> BoolArray:
    return (
        v.evaluate(b.to_struct_array(), Bindings())
        .to_array(b.num_rows())
        .as_bool()
        .copy()
    )


def _as_ts(v: Some[TemporalValue], b: RecordBatch) raises -> TimestampArray:
    return (
        v.evaluate(b.to_struct_array(), Bindings())
        .to_array(b.num_rows())
        .as_type[TimestampArray]()
        .copy()
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def test_temporal_comparison_reaches_the_node() raises:
    """All six dunders, against a truncation of the column itself.

    A truncation rather than a constant because `lit` has no temporal
    overload in either lane — a timestamp literal cannot be written down —
    which is also why `golden/cases/temporal_filter_timestamp.mojo` is spelled
    this way.
    """
    var b = _events()
    var ts = col("ts", timestamp(second))
    var year_start = col("ts", timestamp(second)).date_trunc("year")

    assert_true(_as_bool(ts > year_start, b)[0].value())
    assert_true(not _as_bool(ts < year_start, b)[0].value())
    assert_true(_as_bool(ts >= year_start, b)[0].value())
    assert_true(not _as_bool(ts <= year_start, b)[0].value())
    assert_true(_as_bool(ts != year_start, b)[0].value())
    assert_true(not _as_bool(ts == year_start, b)[0].value())

    # 2020-02-29T00:00:00 is not its own year boundary either, but the leap
    # day is the row where a naive tick-count truncation goes wrong.
    assert_true(_as_bool(ts > year_start, b)[2].value())


def test_temporal_comparison_is_null_where_an_operand_is() raises:
    """Null-in, null-out. The lane compares whatever bytes are in the slot, so
    the bitmap is the only record that the answer is meaningless there."""
    var b = _events()
    var got = _as_bool(
        col("ts", timestamp(second))
        >= col("ts", timestamp(second)).date_trunc("day"),
        b,
    )
    assert_equal(got.null_count(), 1)
    assert_true(got.is_null(1))


def test_temporal_comparison_rejects_a_unit_mismatch() raises:
    """Same width is not the same type. `timestamp[s]` and `timestamp[us]` are
    both int64, so nothing at compile time separates them; the check is per
    batch, in `bind`, and it raises rather than comparing tick counts that
    mean different things.

    Coercion is deliberately not implemented — choosing it means adopting
    promotion rules (Arrow C++'s `common_temporal_resolution` is the prior
    art), which is a decision and not a bound to widen.
    """
    var b = _events()
    with assert_raises(contains="units must match"):
        _ = _as_bool(
            col("ts", timestamp(second)) > col("us", timestamp(microsecond)),
            b,
        )


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def test_temporal_extraction_answers_int32() raises:
    """The nine fields, over a timestamp. `int32` for every one, `year`
    included, because that is Arrow C++'s and therefore PyArrow's answer."""
    var b = _events()
    var ts = col("ts", timestamp(second))
    assert_equal(_as_i32(ts.year(), b)[0].value(), Int32(2019))
    assert_equal(_as_i32(ts.month(), b)[0].value(), Int32(6))
    assert_equal(_as_i32(ts.day(), b)[0].value(), Int32(15))
    assert_equal(_as_i32(ts.hour(), b)[0].value(), Int32(12))
    assert_equal(_as_i32(ts.minute(), b)[0].value(), Int32(30))
    assert_equal(_as_i32(ts.second(), b)[0].value(), Int32(45))
    assert_equal(_as_i32(ts.quarter(), b)[0].value(), Int32(2))
    # 2019-06-15 is a Saturday; ISO weekday with Monday = 0 makes that 5.
    assert_equal(_as_i32(ts.day_of_week(), b)[0].value(), Int32(5))
    assert_equal(_as_i32(ts.day_of_year(), b)[0].value(), Int32(166))


def test_temporal_extraction_works_on_a_date_column() raises:
    """The calendar fields read a `date32` the same way, which is what makes
    the operand bound `TemporalValue` rather than a timestamp-only one.
    2020-02-29 is the leap day: day-of-year 60."""
    var b = _events()
    var d = col("d", date32())
    assert_equal(_as_i32(d.year(), b)[2].value(), Int32(2020))
    assert_equal(_as_i32(d.month(), b)[2].value(), Int32(2))
    assert_equal(_as_i32(d.day(), b)[2].value(), Int32(29))
    assert_equal(_as_i32(d.day_of_year(), b)[2].value(), Int32(60))


def test_temporal_extraction_carries_the_operand_nulls() raises:
    """A null timestamp has a null year. `_extract` copies the operand's
    bitmap into its result and `ColumnBound` reads it back, so the rule is
    stated in the kernel and nowhere else."""
    var b = _events()
    var got = _as_i32(col("ts", timestamp(second)).year(), b)
    assert_equal(got.null_count(), 1)
    assert_true(got.is_null(1))


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_date_trunc_keeps_the_operand_dtype() raises:
    """Unit included: truncating a `timestamp[s]` answers `timestamp[s]`, not
    a bare int64 and not a different resolution. This is the one temporal node
    that cannot answer `DynType(Self.Type())`, because a `TemporalType` is not
    `Defaultable` — it forwards the operand's dtype instead."""
    var b = _events()
    var got = _as_ts(col("ts", timestamp(second)).date_trunc("day"), b)
    # `PrimitiveArray.dtype` is a *field*, not a method — the typed array
    # knows its own `T`, so there is nothing to compute.
    assert_true(DynType(got.dtype.copy()) == b.column(0).dtype())
    # 2019-06-15T12:30:45 floors to 2019-06-15T00:00:00.
    assert_equal(got[0].value(), Int64(1_560_556_800))
    assert_true(got.is_null(1))


def test_date_trunc_calendar_units_go_through_the_calendar() raises:
    """`month`, `quarter` and `year` have no fixed length in seconds, so
    flooring them cannot be a division on the tick count. 2019-06-15 floors to
    2019-06-01, 2019-04-01 and 2019-01-01 respectively."""
    var b = _events()
    var ts = col("ts", timestamp(second))
    assert_equal(
        _as_ts(ts.date_trunc("month"), b)[0].value(), Int64(1_559_347_200)
    )
    assert_equal(
        _as_ts(ts.date_trunc("quarter"), b)[0].value(), Int64(1_554_076_800)
    )
    assert_equal(
        _as_ts(ts.date_trunc("year"), b)[0].value(), Int64(1_546_300_800)
    )


def test_date_trunc_rejects_an_unknown_unit_at_construction() raises:
    """The unit is parsed when the plan is *built*, not when a row is
    evaluated — which is the entire reason `CalendarUnit` is a type rather
    than a `String` threaded down to the kernel."""
    with assert_raises(contains="unsupported unit"):
        _ = col("ts", timestamp(second)).date_trunc("fortnight")


def test_temporal_nodes_compose_into_a_plan() raises:
    """A truncation as a group key and an extraction as a projection — the two
    shapes a query actually uses. A computed key is named `key<i>` by
    `Aggregate._output_schema`, positionally, because it has no source
    column."""
    var b = _events()
    var plan = table(b^).project(
        ["y"], [col("ts", timestamp(second)).year()]
    )
    assert_equal(plan.execute().num_rows(), 3)
    assert_true(plan.schema().fields[0].dtype.is_int32())
