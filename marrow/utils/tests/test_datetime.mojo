# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`CivilDate`, `Epoch` and `floor_div` — the calendar primitives the temporal
kernels and the Parquet INT96 decoder both build on.

These were previously private to `kernels/temporal.mojo` and covered only
indirectly, through whichever extraction kernel happened to exercise them. The
cases here pin the arithmetic itself: the round trip, the pre-epoch branch that
`floor_div` exists for, and the leap-year boundaries where an off-by-one is
invisible on ordinary dates.

Reference values are Hinnant's own worked examples and dates cross-checked
against the proleptic Gregorian calendar Arrow C++ and arrow-rs assume.
"""

from std.testing import assert_equal, assert_false, assert_true

from ..datetime import (
    CivilDate,
    Epoch,
    floor_div,
    Iso8601,
)


def test_civil_date_epoch_is_day_zero() raises:
    """The anchor everything else is relative to."""
    var d = CivilDate.from_days(0)
    assert_true(d == CivilDate(1970, 1, 1))
    assert_equal(d.to_days(), 0)


def test_civil_date_known_dates() raises:
    for c in [
        (1970, 1, 2, 1),
        (1970, 12, 31, 364),
        (1971, 1, 1, 365),
        (2000, 1, 1, 10957),
        (2000, 2, 29, 11016),  # leap day of a 400-divisible year
        (2024, 2, 29, 19782),
        (2038, 1, 19, 24855),  # the 32-bit second overflow date
    ]:
        var expect = CivilDate(c[0], c[1], c[2])
        assert_true(CivilDate.from_days(c[3]) == expect)
        assert_equal(expect.to_days(), c[3])


def test_civil_date_before_the_epoch() raises:
    """The branch `floor_div` exists for: a truncating `//` puts every
    pre-1970 date one era off."""
    assert_true(CivilDate.from_days(-1) == CivilDate(1969, 12, 31))
    assert_equal(CivilDate(1969, 12, 31).to_days(), -1)
    assert_true(CivilDate.from_days(-719162) == CivilDate(1, 1, 1))
    assert_equal(CivilDate(1, 1, 1).to_days(), -719162)


def test_civil_date_round_trips_across_four_centuries() raises:
    """Every day over a full 400-year Gregorian cycle, which is the period of
    the leap rule -- so this covers every case the algorithm has."""
    var bad = 0
    for z in range(-146097, 146097):
        if CivilDate.from_days(z).to_days() != z:
            bad += 1
    assert_equal(bad, 0)


def test_civil_date_day_of_year() raises:
    assert_equal(CivilDate(2021, 1, 1).day_of_year(), 1)
    assert_equal(CivilDate(2021, 12, 31).day_of_year(), 365)
    assert_equal(CivilDate(2020, 12, 31).day_of_year(), 366)  # leap
    assert_equal(CivilDate(2020, 3, 1).day_of_year(), 61)  # after the leap day
    assert_equal(CivilDate(2021, 3, 1).day_of_year(), 60)


def test_civil_date_quarter() raises:
    var got = String()
    for m in range(1, 13):
        got += String(CivilDate(2021, m, 1).quarter())
    assert_equal(got, "111222333444")


def test_civil_date_is_leap() raises:
    assert_true(CivilDate(2020, 1, 1).is_leap())  # divisible by 4
    assert_true(CivilDate(2000, 1, 1).is_leap())  # divisible by 400
    assert_false(CivilDate(1900, 1, 1).is_leap())  # by 100, not 400
    assert_false(CivilDate(2021, 1, 1).is_leap())


def test_civil_date_starts_of_period() raises:
    var d = CivilDate(2021, 8, 17)
    assert_true(d.start_of_month() == CivilDate(2021, 8, 1))
    assert_true(d.start_of_quarter() == CivilDate(2021, 7, 1))
    assert_true(d.start_of_year() == CivilDate(2021, 1, 1))


def test_civil_date_start_of_quarter_for_every_month() raises:
    var got = String()
    for m in range(1, 13):
        got += String(CivilDate(2021, m, 28).start_of_quarter().month)
        got += ","
    assert_equal(got, "1,1,1,4,4,4,7,7,7,10,10,10,")


def test_civil_date_writes_iso_8601() raises:
    assert_equal(String(CivilDate(2021, 8, 17)), "2021-08-17")
    assert_equal(String(CivilDate(2021, 1, 2)), "2021-01-02")
    assert_equal(String(CivilDate(33, 1, 2)), "0033-01-02")
    assert_equal(String(CivilDate(-1, 12, 31)), "-1-12-31")


def test_floor_div_rounds_toward_negative_infinity() raises:
    assert_equal(floor_div(7, 3), 2)
    assert_equal(floor_div(-7, 3), -3)  # truncating `//` would give -2
    assert_equal(floor_div(-1, 400), -1)
    assert_equal(floor_div(0, 400), 0)
    assert_equal(floor_div(-400, 400), -1)


def test_epoch_constants_are_consistent() raises:
    assert_equal(Epoch.MILLIS_PER_DAY, Epoch.SECONDS_PER_DAY * 1_000)
    assert_equal(Epoch.MICROS_PER_DAY, Epoch.SECONDS_PER_DAY * 1_000_000)
    assert_equal(Epoch.NANOS_PER_DAY, Epoch.SECONDS_PER_DAY * 1_000_000_000)


def test_epoch_julian_day_matches_the_civil_epoch() raises:
    """Parquet INT96 stores a Julian day number; the reader converts by
    subtracting this constant, so it has to agree with day zero."""
    assert_equal(Epoch.JULIAN_DAY, 2440588)
    assert_equal(CivilDate.from_days(0).to_days() + Epoch.JULIAN_DAY, 2440588)


def test_civil_date_days_in_month_and_validity() raises:
    """Leap years by all three rules, and the month and day bounds."""
    assert_equal(CivilDate(2000, 2, 1).days_in_month(), 29)
    assert_equal(CivilDate(1900, 2, 1).days_in_month(), 28)
    assert_equal(CivilDate(2024, 2, 1).days_in_month(), 29)
    assert_equal(CivilDate(2023, 4, 1).days_in_month(), 30)
    assert_equal(CivilDate(2023, 12, 1).days_in_month(), 31)
    assert_true(CivilDate(2024, 2, 29).is_valid())
    assert_false(CivilDate(2023, 2, 29).is_valid())
    assert_false(CivilDate(2023, 0, 1).is_valid())
    assert_false(CivilDate(2023, 13, 1).is_valid())
    assert_false(CivilDate(2023, 4, 31).is_valid())
    assert_false(CivilDate(2023, 1, 0).is_valid())


def _iso[digits: Int](text: String) -> Optional[Int]:
    return Iso8601[digits].parse(text.as_bytes())


def test_iso8601_parse_dates() raises:
    """Arrow C++'s `ToTimestampDate_ISO8601` cases, second unit."""
    for c in [
        ("1970-01-01", 0),
        ("1989-07-14", 616377600),
        ("2000-02-29", 951782400),
        ("3989-07-14", 63730281600),
        ("1900-02-28", -2203977600),
    ]:
        assert_equal(_iso[0](c[0]).value(), c[1])
    for text in [
        "",
        "1970",
        "19700101",
        "1970/01/01",
        "1970-01-01 ",
        "1970-01-01Z",
        "1970-00-01",
        "1970-13-01",
        "1970-01-32",
        "1970-02-29",
        "2100-02-29",
    ]:
        assert_false(Bool(_iso[0](text)), text)


def test_iso8601_parse_datetimes() raises:
    """Arrow C++'s `ToTimestampDateTime_ISO8601` cases, second unit: every
    clock precision crossed with every zone-offset spelling."""
    for c in [
        ("1970-01-01 00:00:00", 0),
        ("2018-11-13 17", 1542128400),
        ("2018-11-13 17+00", 1542128400),
        ("2018-11-13 17+0000", 1542128400),
        ("2018-11-13 17+00:00", 1542128400),
        ("2018-11-13 17+01", 1542124800),
        ("2018-11-13 17+0117", 1542123780),
        ("2018-11-13 17+01:17", 1542123780),
        ("2018-11-13 17-01", 1542132000),
        ("2018-11-13 17-0117", 1542133020),
        ("2018-11-13 17-01:17", 1542133020),
        ("2018-11-13T17", 1542128400),
        ("2018-11-13 17Z", 1542128400),
        ("2018-11-13T17:11", 1542129060),
        ("2018-11-13 17:11Z", 1542129060),
        ("2018-11-13 17:11+01:17", 1542124440),
        ("2018-11-13 17:11-0117", 1542133680),
        ("2018-11-13T17:11:10", 1542129070),
        ("2018-11-13T17:11:10Z", 1542129070),
        ("2018-11-13T17:11:10+01", 1542125470),
        ("2018-11-13T17:11:10-01:17", 1542133690),
        ("1900-02-28 12:34:56", -2203932304),
    ]:
        assert_equal(_iso[0](c[0]).value(), c[1], c[0])
    for text in [
        "1900-02-28 12:34:56.001",
        "1970-02-29 00:00:00",
        "1970-01-01 24",
        "1970-01-01 00:60",
        "1970-01-01 00,00",
        "1970-01-01 24:00:00",
        "1970-01-01 00:00:60",
        "1970-01-01 00:00,00",
        "1970-01-01 00:00+0",
        "1970-01-01 00:00+000",
        "1970-01-01 00:00+00000",
        "1970-01-01 00:00+2400",
        "1970-01-01 00:00+0060",
        "1970-01-01 00-0",
        "1970-01-01 00+00000",
        "1970-01-01 00:00:00-000",
        "1970-01-01 00:00:00+00:99",
    ]:
        assert_false(Bool(_iso[0](text)), text)


def test_iso8601_parse_subseconds() raises:
    """Fractional seconds pad to the unit and are rejected past it."""
    assert_equal(_iso[3]("2018-11-13T17:11:10.777Z").value(), 1542129070777)
    assert_equal(_iso[3]("1900-02-28 12:34:56.1").value(), -2203932304000 + 100)
    assert_equal(
        _iso[3]("2018-11-13 17:11:10.123+01:17").value(),
        1542129070123 - 4620000,
    )
    assert_false(Bool(_iso[3]("1900-02-28 12:34:56.1234")))
    assert_equal(
        _iso[6]("3989-07-14T11:22:33.000777Z").value(), 63730322553000777
    )
    assert_equal(
        _iso[9]("1900-02-28 12:34:56.123456789").value(),
        -2203932304000000000 + 123456789,
    )
    assert_false(Bool(_iso[9]("1900-02-28 12:34:56.1234567890")))


def test_iso8601_parse_out_of_range_for_the_unit() raises:
    """Nanoseconds since the epoch cover only ~1677-2262; a date outside that
    range is `None` rather than a wrapped value."""
    assert_equal(_iso[0]("3989-07-14").value(), 63730281600)
    assert_false(Bool(_iso[9]("3989-07-14")))
    assert_false(Bool(_iso[9]("1600-01-01")))


def test_iso8601_parse_fraction_cannot_carry_past_the_maximum() raises:
    """At the last whole second int64 nanoseconds can hold, the fraction
    decides: up to `.854775807` fits, one more would wrap."""
    assert_equal(
        _iso[9]("2262-04-11 23:47:16.854775807").value(), Int(Int64.MAX)
    )
    assert_false(Bool(_iso[9]("2262-04-11 23:47:16.854775808")))


def _written[digits: Int](ticks: Int) -> String:
    var out = String()
    Iso8601[digits].write(ticks, out)
    return out^


def test_iso8601_write_is_the_inverse_of_parse() raises:
    """Before and after the epoch, with and without a fraction: what is
    written parses back to the same ticks."""
    assert_equal(_written[0](0), "1970-01-01 00:00:00")
    assert_equal(_written[0](-2203932304), "1900-02-28 12:34:56")
    assert_equal(_written[3](1542129070777), "2018-11-13 17:11:10.777")
    assert_equal(_written[3](1542129070000), "2018-11-13 17:11:10")
    assert_equal(_written[9](-1), "1969-12-31 23:59:59.999999999")
    for ticks in [0, -1, 1, 1542129070777, -2203932304000, 63730322553000]:
        var text = _written[3](ticks)
        assert_equal(_iso[3](text).value(), ticks, text)


def test_iso8601_write_pads_small_years() raises:
    var ticks = CivilDate(33, 1, 2).to_days() * Epoch.SECONDS_PER_DAY
    assert_equal(_written[0](ticks), "0033-01-02 00:00:00")
    assert_equal(_iso[0](_written[0](ticks)).value(), ticks)
