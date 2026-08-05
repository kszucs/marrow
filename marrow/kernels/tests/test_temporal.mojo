"""Unit tests for the temporal compute kernels (marrow.kernels.temporal)."""

from std.testing import (
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
)

from ...arrays import (
    DynArray,
    Int32Array,
    Date32Array,
    Date64Array,
    Time32Array,
    Time64Array,
    TimestampArray,
    PrimitiveArray,
)
from ...builders import PrimitiveBuilder, array
from ...dtypes import (
    DurationType,
    duration,
    int32,
    second,
    date32,
    date64,
    time32,
    time64,
    timestamp,
    millisecond,
    microsecond,
    nanosecond,
    Date32Type,
    Date64Type,
    Time32Type,
    Time64Type,
    TimestampType,
)
from ...kernels.temporal import (
    YearKernel,
    MonthKernel,
    DayKernel,
    HourKernel,
    MinuteKernel,
    SecondKernel,
    QuarterKernel,
    DayOfYearKernel,
    DayOfWeekKernel,
    DateTruncKernel,
    CalendarUnit,
    unit_month,
    unit_quarter,
    unit_year,
    CalendarUnit,
    unit_second,
    unit_minute,
    unit_hour,
    unit_day,
)


# --- helpers ---------------------------------------------------------------


def _ts(values: List[Int], unit: TimestampType) raises -> TimestampArray:
    var b = PrimitiveBuilder[TimestampType](unit, capacity=len(values))
    for v in values:
        b.append(Int64(v))
    return b.finish()


def _ts_with_null() raises -> TimestampArray:
    # 2019-06-15 12:30:45 UTC = 1560601845 s ; null ; 2020-02-29 00:00:00 UTC
    var b = PrimitiveBuilder[TimestampType](timestamp(second), capacity=3)
    b.append(Int64(1_560_601_845))
    b.append_null()
    b.append(Int64(1_582_934_400))
    return b.finish()


def _d32(values: List[Int]) raises -> Date32Array:
    var b = PrimitiveBuilder[Date32Type](date32(), capacity=len(values))
    for v in values:
        b.append(Int32(v))
    return b.finish()


def _t64(values: List[Int], unit: Time64Type) raises -> Time64Array:
    var b = PrimitiveBuilder[Time64Type](unit, capacity=len(values))
    for v in values:
        b.append(Int64(v))
    return b.finish()


# --- year / month / day (timestamp) ---------------------------------------


def test_year_month_day_timestamp() raises:
    # 2019-06-15 12:30:45, 2020-02-29 00:00:00, 1970-01-01 00:00:00
    var a = _ts([1_560_601_845, 1_582_934_400, 0], timestamp(second))
    assert_true(YearKernel.apply(a) == array([2019, 2020, 1970], int32))
    assert_true(MonthKernel.apply(a) == array([6, 2, 1], int32))
    assert_true(DayKernel.apply(a) == array([15, 29, 1], int32))


def test_hour_minute_second_timestamp() raises:
    # 2019-06-15 12:30:45 UTC
    var a = _ts([1_560_601_845], timestamp(second))
    assert_true(HourKernel.apply(a) == array([12], int32))
    assert_true(MinuteKernel.apply(a) == array([30], int32))
    assert_true(SecondKernel.apply(a) == array([45], int32))


def test_subsecond_units_truncate_to_second() raises:
    # 1_560_601_845_123 ms == 2019-06-15 12:30:45.123 UTC
    var a = _ts([1_560_601_845_123], timestamp(millisecond))
    assert_true(YearKernel.apply(a) == array([2019], int32))
    assert_true(HourKernel.apply(a) == array([12], int32))
    assert_true(SecondKernel.apply(a) == array([45], int32))


def test_quarter() raises:
    # Jan, Apr, Jul, Oct, Dec 2021
    var a = _ts(
        [
            1_609_459_200,  # 2021-01-01
            1_617_235_200,  # 2021-04-01
            1_625_097_600,  # 2021-07-01
            1_633_046_400,  # 2021-10-01
            1_638_316_800,  # 2021-12-01
        ],
        timestamp(second),
    )
    assert_true(QuarterKernel.apply(a) == array([1, 2, 3, 4, 4], int32))


def test_day_of_week() raises:
    # 2019-01-01 Tue, 2019-01-05 Sat, 2019-01-06 Sun, 1970-01-01 Thu
    var a = _ts(
        [1_546_300_800, 1_546_646_400, 1_546_732_800, 0], timestamp(second)
    )
    # ISO weekday, Monday=0: Tue=1, Sat=5, Sun=6, Thu=3
    assert_true(DayOfWeekKernel.apply(a) == array([1, 5, 6, 3], int32))


def test_day_of_year() raises:
    # 2019-01-01 -> 1, 2020-12-31 (leap) -> 366, 2019-03-01 -> 60
    var a = _ts(
        [1_546_300_800, 1_609_372_800, 1_551_398_400], timestamp(second)
    )
    assert_true(DayOfYearKernel.apply(a) == array([1, 366, 60], int32))


# --- date32 ----------------------------------------------------------------


def test_extract_date32() raises:
    # days since epoch: 0 = 1970-01-01, 17897 = 2019-01-01, 18628 = 2021-01-01
    var a = _d32([0, 17897, 18628])
    assert_true(
        YearKernel.dispatch(a.copy()) == array([1970, 2019, 2021], int32)
    )
    assert_true(MonthKernel.dispatch(a.copy()) == array([1, 1, 1], int32))
    assert_true(DayKernel.dispatch(a.copy()) == array([1, 1, 1], int32))


# --- time64 (clock only) ---------------------------------------------------


def test_extract_time64() raises:
    # 12:30:45.000000 in microseconds since midnight
    var micros = ((12 * 3600 + 30 * 60 + 45)) * 1_000_000
    var a = _t64([micros], time64(microsecond))
    assert_true(HourKernel.apply(a) == array([12], int32))
    assert_true(MinuteKernel.apply(a) == array([30], int32))
    assert_true(SecondKernel.apply(a) == array([45], int32))


def test_calendar_field_on_time_raises() raises:
    var micros = 1_000_000
    var a = _t64([micros], time64(microsecond))
    with assert_raises():
        _ = YearKernel.dispatch(a.copy().to_dyn())


# --- null propagation ------------------------------------------------------


def test_null_propagation() raises:
    var r = MinuteKernel.apply(_ts_with_null())
    assert_equal(r.null_count(), 1)
    assert_true(r.is_valid(0))
    assert_false(r.is_valid(1))
    assert_true(r.is_valid(2))
    assert_equal(r[0].value(), 30)
    assert_equal(r[2].value(), 0)


# --- pre-epoch (negative) --------------------------------------------------


def test_pre_epoch_timestamp() raises:
    # 1969-12-31 23:59:59 UTC = -1 s
    var a = _ts([-1], timestamp(second))
    assert_true(YearKernel.apply(a) == array([1969], int32))
    assert_true(MonthKernel.apply(a) == array([12], int32))
    assert_true(DayKernel.apply(a) == array([31], int32))
    assert_true(HourKernel.apply(a) == array([23], int32))
    assert_true(MinuteKernel.apply(a) == array([59], int32))
    assert_true(SecondKernel.apply(a) == array([59], int32))


# --- date_trunc ------------------------------------------------------------


def test_date_trunc_minute() raises:
    # 2019-06-15 12:30:45 -> 2019-06-15 12:30:00 = 1560601800
    var a = _ts([1_560_601_845], timestamp(second))
    var r = DateTruncKernel.apply(a.copy().to_dyn(), unit_minute)
    assert_true(r.dtype() == timestamp(second).to_dyn())
    assert_equal(r.as_timestamp()[0].value(), 1_560_601_800)


def test_date_trunc_units() raises:
    var a = _ts([1_560_601_845], timestamp(second))  # 2019-06-15 12:30:45
    assert_equal(
        DateTruncKernel.apply(a.copy().to_dyn(), unit_second)
        .as_timestamp()[0]
        .value(),
        1_560_601_845,
    )
    assert_equal(
        DateTruncKernel.apply(a.copy().to_dyn(), unit_minute)
        .as_timestamp()[0]
        .value(),
        1_560_601_800,
    )
    assert_equal(
        DateTruncKernel.apply(a.copy().to_dyn(), unit_hour)
        .as_timestamp()[0]
        .value(),
        1_560_600_000,  # 2019-06-15 12:00:00
    )
    assert_equal(
        DateTruncKernel.apply(a.copy().to_dyn(), unit_day)
        .as_timestamp()[0]
        .value(),
        1_560_556_800,  # 2019-06-15 00:00:00
    )


def test_date_trunc_millisecond_unit() raises:
    # 2019-06-15 12:30:45.123 ms -> minute -> 2019-06-15 12:30:00.000
    var a = _ts([1_560_601_845_123], timestamp(millisecond))
    var r = DateTruncKernel.apply(a.copy().to_dyn(), unit_minute)
    assert_equal(r.as_timestamp()[0].value(), 1_560_601_800_000)


def test_date_trunc_preserves_nulls() raises:
    var r = DateTruncKernel.apply(_ts_with_null().to_dyn(), unit_hour)
    assert_equal(r.null_count(), 1)
    assert_false(r.is_valid(1))
    assert_equal(r.as_timestamp()[0].value(), 1_560_600_000)


# --- cross-check against PyArrow -------------------------------------------


def test_cross_check_pyarrow() raises:
    from std.python import Python

    var pa = Python.import_module("pyarrow")
    var pc = Python.import_module("pyarrow.compute")

    # A spread of second-resolution UTC timestamps.
    var raw = [
        0,
        1_560_601_845,
        1_582_934_400,
        1_609_459_200,
        1_546_646_400,
        -1,
        915_148_800,  # 1999-01-01
    ]
    var a = _ts(raw, timestamp(second))
    var pylist = Python.list()
    for v in raw:
        pylist.append(v)
    var pa_arr = pa.array(pylist, type=pa.timestamp("s"))

    var yr = YearKernel.apply(a)
    var mo = MonthKernel.apply(a)
    var dy = DayKernel.apply(a)
    var hr = HourKernel.apply(a)
    var mi = MinuteKernel.apply(a)
    var se = SecondKernel.apply(a)
    var qt = QuarterKernel.apply(a)
    var doy = DayOfYearKernel.apply(a)
    var dow = DayOfWeekKernel.apply(a)

    var pa_yr = pc.year(pa_arr)
    var pa_mo = pc.month(pa_arr)
    var pa_dy = pc.day(pa_arr)
    var pa_hr = pc.hour(pa_arr)
    var pa_mi = pc.minute(pa_arr)
    var pa_se = pc.second(pa_arr)
    var pa_qt = pc.quarter(pa_arr)
    var pa_doy = pc.day_of_year(pa_arr)
    var pa_dow = pc.day_of_week(pa_arr)

    for i in range(len(raw)):
        assert_equal(Int(yr[i].value()), Int(py=pa_yr[i]))
        assert_equal(Int(mo[i].value()), Int(py=pa_mo[i]))
        assert_equal(Int(dy[i].value()), Int(py=pa_dy[i]))
        assert_equal(Int(hr[i].value()), Int(py=pa_hr[i]))
        assert_equal(Int(mi[i].value()), Int(py=pa_mi[i]))
        assert_equal(Int(se[i].value()), Int(py=pa_se[i]))
        assert_equal(Int(qt[i].value()), Int(py=pa_qt[i]))
        assert_equal(Int(doy[i].value()), Int(py=pa_doy[i]))
        assert_equal(Int(dow[i].value()), Int(py=pa_dow[i]))


def test_cross_check_date_trunc_pyarrow() raises:
    from std.python import Python

    var pa = Python.import_module("pyarrow")
    var pc = Python.import_module("pyarrow.compute")

    var raw = [1_560_601_845, 1_582_934_400, 0, 915_148_800]
    var a = _ts(raw, timestamp(second))
    var pylist = Python.list()
    for v in raw:
        pylist.append(v)
    var pa_arr = pa.array(pylist, type=pa.timestamp("s"))

    for unit in ["second", "minute", "hour", "day"]:
        var r = DateTruncKernel.apply(
            a.copy().to_dyn(), CalendarUnit.parse(unit)
        )
        var pa_r = pc.floor_temporal(pa_arr, unit=unit)
        for i in range(len(raw)):
            assert_equal(
                Int(r.as_timestamp()[i].value()), Int(py=pa_r[i].value)
            )


def test_date_trunc_duration() raises:
    """`date_trunc` accepts a duration, which the old five-arm ladder rejected.

    `DurationType` is a `TemporalType`, and `_ticks_per_second` always handled
    it -- only the hand-written ladder had forgotten it. Walking the family
    closed that gap, so this pins the behaviour rather than leaving it an
    untested side effect: 3661 seconds floored to the hour is 3600.
    """
    var b = PrimitiveBuilder[DurationType](duration(second), capacity=2)
    b.append(Int64(3661))
    b.append(Int64(59))
    var d = b.finish()

    var r = DateTruncKernel.apply(d.copy().to_dyn(), unit_hour)
    assert_equal(Int(r.as_primitive[DurationType]()[0].value()), 3600)
    assert_equal(Int(r.as_primitive[DurationType]()[1].value()), 0)


def test_extract_rejects_duration_with_a_useful_message() raises:
    """A duration reaches the extraction kernels now, and is refused by
    `_extract` naming the real reason rather than by the dispatcher claiming it
    is not temporal."""
    var b = PrimitiveBuilder[DurationType](duration(second), capacity=1)
    b.append(Int64(3661))
    var d = b.finish()

    with assert_raises(contains="requires a date or timestamp array"):
        _ = YearKernel.dispatch(d.copy().to_dyn())


# --- date_trunc, calendar units (M1.4) --------------------------------------
#
# second/minute/hour/day are fixed-length, so flooring is one division on the
# tick count. month/quarter/year are not: a month is 28-31 days, so these have
# to go through the civil calendar. ClickBench Q35/Q36 group by month, and
# without them they fail as queries rather than as compile errors.


def test_date_trunc_month() raises:
    # 2019-06-15 12:30:45 -> 2019-06-01 00:00:00 = 1559347200
    var a = _ts([1_560_601_845], timestamp(second))
    var r = DateTruncKernel.apply(a.copy().to_dyn(), unit_month)
    assert_true(r.dtype() == timestamp(second).to_dyn())
    assert_equal(r.as_timestamp()[0].value(), 1_559_347_200)


def test_date_trunc_quarter() raises:
    # 2019-06-15 is Q2, which starts 2019-04-01 00:00:00 = 1554076800
    var a = _ts([1_560_601_845], timestamp(second))
    var r = DateTruncKernel.apply(a.copy().to_dyn(), unit_quarter)
    assert_equal(r.as_timestamp()[0].value(), 1_554_076_800)


def test_date_trunc_year() raises:
    # 2019-01-01 00:00:00 = 1546300800
    var a = _ts([1_560_601_845], timestamp(second))
    var r = DateTruncKernel.apply(a.copy().to_dyn(), unit_year)
    assert_equal(r.as_timestamp()[0].value(), 1_546_300_800)


def test_date_trunc_quarter_boundaries() raises:
    """Each quarter floors to its own first month, not to the year."""
    # 2019-01-10, 2019-05-10, 2019-08-10, 2019-11-10
    var a = _ts(
        [1_547_078_400, 1_557_446_400, 1_565_395_200, 1_573_344_000],
        timestamp(second),
    )
    var r = DateTruncKernel.apply(a.copy().to_dyn(), unit_quarter)
    ref t = r.as_timestamp()
    assert_equal(t[0].value(), 1_546_300_800)  # Q1 -> 2019-01-01
    assert_equal(t[1].value(), 1_554_076_800)  # Q2 -> 2019-04-01
    assert_equal(t[2].value(), 1_561_939_200)  # Q3 -> 2019-07-01
    assert_equal(t[3].value(), 1_569_888_000)  # Q4 -> 2019-10-01


def test_date_trunc_calendar_units_apply_to_date32() raises:
    """`date32` is day-granular, so sub-day units are a no-op — but month, quarter
    and year are *not*, and the early return that skipped them was wrong."""
    var db = PrimitiveBuilder[Date32Type](date32(), capacity=1)
    db.append(Int32(18_062))  # 2019-06-15
    var d = db.finish()
    var m = DateTruncKernel.apply(d.copy().to_dyn(), unit_month)
    assert_true(m.dtype() == date32().to_dyn())
    assert_equal(m.as_date32()[0].value(), 18_048)  # 2019-06-01
    var y = DateTruncKernel.apply(d.copy().to_dyn(), unit_year)
    assert_equal(y.as_date32()[0].value(), 17_897)  # 2019-01-01
    # sub-day still a no-op
    assert_equal(
        DateTruncKernel.apply(d.copy().to_dyn(), unit_hour)
        .as_date32()[0]
        .value(),
        18_062,
    )


def test_calendar_unit_parses_the_new_names() raises:
    assert_true(CalendarUnit.parse("month") == unit_month)
    assert_true(CalendarUnit.parse("quarter") == unit_quarter)
    assert_true(CalendarUnit.parse("year") == unit_year)
    assert_equal(String(unit_quarter), "quarter")
