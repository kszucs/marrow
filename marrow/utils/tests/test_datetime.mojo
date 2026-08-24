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

from ..datetime import CivilDate, Epoch, floor_div


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
