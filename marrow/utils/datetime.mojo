# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Proleptic-Gregorian calendar arithmetic, and time zones on top of it.

Shared by `kernels/temporal.mojo`, `kernels/cast.mojo` and
`parquet/reader.mojo` without any of them depending on each other. The calendar
half imports nothing; `WallClock` and `TimestampFormat` add the tz database
(morrow) and marrow's error kinds, nothing above them.

Everything here is integer arithmetic on a **day count** — days since
1970-01-01, the unit every Arrow date and timestamp reduces to. The
*resolution* lookups (`TimeUnit` -> ticks per second, dtype -> nanoseconds per
tick) stay in `kernels/temporal.mojo`: they are keyed on `marrow.dtypes`
types, which this module does not import.

`WallClock` turns a zoned timestamp's UTC ticks into wall-clock ticks and
back; zones come from morrow, which reads the system tz database once per zone
per process and resolves a skipped wall-clock time with the offset before it
and a repeated one as its earlier instant, as ICU and DuckDB do.
`TimestampFormat` writes one as text.

**Shaped for hot loops.** `CivilDate` is three `Int`s with no heap state and no
validity, so it is register-passable and trivially copyable; every method is
`@always_inline`. The extraction kernels call these once per element, so a
non-inlined call or a heap allocation here would show up as a per-row cost.
"""


from morrow import TimeZone

from ..errors import InvalidError, IOError, KeyError


@always_inline
def floor_div(a: Int, b: Int) -> Int:
    """Floor division for a positive divisor `b`, independent of `//`'s
    rounding.

    Mojo's `//` truncates toward zero for `Int`, so `-1 // 400` is 0 where the
    civil-date algorithms need -1. Every pre-epoch date depends on this.
    """
    var q = a // b
    if a - q * b < 0:
        q -= 1
    return q


@always_inline
def split_days[
    W: Int
](ticks: SIMD[DType.int64, W], per_day: Int64) -> Tuple[
    SIMD[DType.int64, W], SIMD[DType.int64, W]
]:
    """Ticks since the epoch, `per_day` of them to a day, as the day they fall
    in and the ticks into that day.

    The day is floored, so the time of day is never negative: one second
    before the epoch is 23:59:59 on 1969-12-31.
    """
    var days = ticks // per_day
    return (days, ticks - days * per_day)


@always_inline
def truncate_div[
    W: Int
](ticks: SIMD[DType.int64, W], divisor: Int64) -> SIMD[DType.int64, W]:
    """Ticks divided by a positive `divisor`, truncated toward zero, which is
    how Arrow converts a tick to a coarser unit: -1500 ms is -1 s, not -2.
    """
    var q = ticks // divisor
    var r = ticks - q * divisor
    return q + (r.ne(0) & ticks.lt(0)).cast[DType.int64]()


struct Epoch:
    """Unix-epoch constants. A namespace, never instantiated.

    `JULIAN_DAY` is here because Parquet's INT96 timestamps are Julian-day
    based and `parquet/reader.mojo` had its own copy of it.
    """

    comptime JULIAN_DAY = 2440588
    """Julian day number of 1970-01-01."""

    comptime SECONDS_PER_DAY = 86_400
    comptime MILLIS_PER_DAY = 86_400_000
    comptime MICROS_PER_DAY = 86_400_000_000
    comptime NANOS_PER_DAY = 86_400_000_000_000


struct CivilDate(Copyable, Equatable, ImplicitlyCopyable, Movable, Writable):
    """A proleptic-Gregorian date, decomposed into year, month and day.

    Howard Hinnant's `civil_from_days` / `days_from_civil`, which is what Arrow
    C++ and arrow-rs both use, so date extraction agrees with them by
    construction rather than by coincidence. The algorithms are exact for the
    whole `Int` range and shift the era to start in March, which is what makes
    the leap-day the *last* day of the year and removes every special case.

    Held as a struct rather than returned as `Tuple[Int, Int, Int]` because
    every caller indexed that tuple positionally -- `c[0]`, `c[1]`, `c[2]` --
    and one of them had to spell its own inverse to get the day of the year.
    """

    var year: Int
    var month: Int
    """1-12."""
    var day: Int
    """1-31."""

    @always_inline
    def __init__(out self, year: Int, month: Int, day: Int):
        self.year = year
        self.month = month
        self.day = day

    @staticmethod
    @always_inline
    def from_days(z: Int) -> Self:
        """Days since 1970-01-01 -> a civil date."""
        var zz = z + 719468
        var era = floor_div(zz, 146097)
        var doe = zz - era * 146097  # [0, 146096]
        var yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
        var y = yoe + era * 400
        var doy = doe - (365 * yoe + yoe // 4 - yoe // 100)  # [0, 365]
        var mp = (5 * doy + 2) // 153  # [0, 11]
        var d = doy - (153 * mp + 2) // 5 + 1  # [1, 31]
        var m = mp + 3 if mp < 10 else mp - 9  # [1, 12]
        return Self(y + 1 if m <= 2 else y, m, d)

    @staticmethod
    @always_inline
    def days_from(year: Int, month: Int, day: Int) -> Int:
        """`(y, m, d)` -> days since 1970-01-01, without building a `CivilDate`.

        The static form exists so `day_of_year` can ask for January 1st of its
        own year without constructing a temporary for it.
        """
        var yy = year - 1 if month <= 2 else year
        var era = floor_div(yy, 400)
        var yoe = yy - era * 400  # [0, 399]
        var mp = month - 3 if month > 2 else month + 9
        var doy = (153 * mp + 2) // 5 + day - 1  # [0, 365]
        var doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
        return era * 146097 + doe - 719468

    @always_inline
    def to_days(self) -> Int:
        """Days since 1970-01-01. Exact inverse of `from_days`."""
        return Self.days_from(self.year, self.month, self.day)

    @always_inline
    def day_of_year(self) -> Int:
        """1 on January 1st, 365 or 366 on December 31st."""
        return self.to_days() - Self.days_from(self.year, 1, 1) + 1

    @always_inline
    def quarter(self) -> Int:
        """1-4."""
        return (self.month - 1) // 3 + 1

    @always_inline
    def is_leap(self) -> Bool:
        var y = self.year
        return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)

    @always_inline
    def days_in_month(self) -> Int:
        """28-31."""
        if self.month == 2:
            return 29 if self.is_leap() else 28
        var m = self.month
        return 30 if m == 4 or m == 6 or m == 9 or m == 11 else 31

    @always_inline
    def is_valid(self) -> Bool:
        """Whether this names a real day: a month of 1-12, and a day that
        month has."""
        return (
            self.month >= 1
            and self.month <= 12
            and self.day >= 1
            and self.day <= self.days_in_month()
        )

    @always_inline
    def start_of_year(self) -> Self:
        return Self(self.year, 1, 1)

    @always_inline
    def start_of_quarter(self) -> Self:
        # 1-3 -> 1, 4-6 -> 4, 7-9 -> 7, 10-12 -> 10.
        return Self(self.year, ((self.month - 1) // 3) * 3 + 1, 1)

    @always_inline
    def start_of_month(self) -> Self:
        return Self(self.year, self.month, 1)

    @always_inline
    def end_of_month(self) -> Self:
        return Self(self.year, self.month, self.days_in_month())

    def month_name(self) -> StaticString:
        """The full English name of this date's month."""
        var m = self.month
        if m == 1:
            return "January"
        elif m == 2:
            return "February"
        elif m == 3:
            return "March"
        elif m == 4:
            return "April"
        elif m == 5:
            return "May"
        elif m == 6:
            return "June"
        elif m == 7:
            return "July"
        elif m == 8:
            return "August"
        elif m == 9:
            return "September"
        elif m == 10:
            return "October"
        elif m == 11:
            return "November"
        else:
            return "December"

    def __eq__(self, other: Self) -> Bool:
        return (
            self.year == other.year
            and self.month == other.month
            and self.day == other.day
        )

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    def write_to[W: Writer](self, mut writer: W):
        """`YYYY-MM-DD`. A year outside 0-9999 has no four-digit form and is
        written as its plain number."""
        if self.year >= 0 and self.year <= 9999:
            _write_padded(writer, self.year, 4)
        else:
            writer.write(self.year)
        writer.write("-")
        _write_padded(writer, self.month, 2)
        writer.write("-")
        _write_padded(writer, self.day, 2)


@always_inline
def weekday(days: Int) -> Int:
    """The weekday of a days-since-epoch value, Monday = 0 ... Sunday = 6."""
    # 1970-01-01 was a Thursday.
    return days + 3 - floor_div(days + 3, 7) * 7


@always_inline
def iso_week_thursday(days: Int) -> Int:
    """Days-since-epoch of the Thursday of the ISO week holding `days` — the
    day whose calendar year is that week's ISO year."""
    return days - weekday(days) + 3


def weekday_name(weekday: Int) -> StaticString:
    """The full English name of `weekday`, Monday = 0."""
    if weekday == 0:
        return "Monday"
    elif weekday == 1:
        return "Tuesday"
    elif weekday == 2:
        return "Wednesday"
    elif weekday == 3:
        return "Thursday"
    elif weekday == 4:
        return "Friday"
    elif weekday == 5:
        return "Saturday"
    else:
        return "Sunday"


struct CalendarUnit(Equatable, ImplicitlyCopyable, Movable, Writable):
    """The boundary `date_trunc` floors to — `second`, `minute`, `hour`, `day`.

    A *calendar* unit, not a storage resolution: distinct from `dtypes.TimeUnit`
    (`s`/`ms`/`us`/`ns`), which says how finely a timestamp is stored. Named
    after Arrow C++'s `CalendarUnit`, which plays the same role in
    `RoundTemporalOptions`.

    This is a type rather than a `String` so an unsupported unit cannot reach
    the kernel at all. String-driven callers (Python, SQL) convert once at their
    boundary with `parse`, which is the only place the spelling is known — the
    expression frontends do it at *construction*, so a bad unit fails when the
    plan is built rather than on the row that first evaluates it.
    """

    var value: Int

    def __init__(out self, value: Int):
        self.value = value

    @staticmethod
    def parse(unit: String) raises -> Self:
        """The unit named by `unit`; raises if it names none."""
        if unit == "second":
            return Self(0)
        elif unit == "minute":
            return Self(1)
        elif unit == "hour":
            return Self(2)
        elif unit == "day":
            return Self(3)
        elif unit == "month":
            return Self(4)
        elif unit == "quarter":
            return Self(5)
        elif unit == "year":
            return Self(6)
        raise InvalidError(
            t"date_trunc: unsupported unit '{unit}' (expected "
            t"second/minute/hour/day/month/quarter/year)"
        )

    def is_calendar(self) -> Bool:
        """Whether this unit has no fixed length in seconds.

        A month is 28-31 days and a year 365 or 366, so `seconds()` cannot
        answer for these and flooring cannot be a division on the tick count --
        they go through the civil calendar instead. Everything up to `day` is
        fixed-length.
        """
        return self.value >= 4

    def seconds(self) -> Int:
        """How many seconds one of these spans."""
        if self.value == 0:
            return 1
        elif self.value == 1:
            return 60
        elif self.value == 2:
            return 3600
        else:
            return 86400

    def floor(self, ticks: Int, ticks_per_day: Int) -> Int:
        """`ticks` floored to the start of the unit it falls in, at a
        resolution of `ticks_per_day` ticks a day — 1 for a day count.

        Floors rather than truncates, so a pre-epoch tick lands in the unit
        that contains it rather than the one after. A unit finer than the
        resolution leaves `ticks` as it is."""
        if self.is_calendar():
            var c = CivilDate.from_days(floor_div(ticks, ticks_per_day))
            var start: CivilDate
            if self.value == 6:
                start = c.start_of_year()
            elif self.value == 5:
                start = c.start_of_quarter()
            else:
                start = c.start_of_month()
            return start.to_days() * ticks_per_day
        var per_unit = max(ticks_per_day // 86400 * self.seconds(), 1)
        return floor_div(ticks, per_unit) * per_unit

    def to_string(self) -> StaticString:
        """The unit's spelling. `StaticString`, not `String`: these are four
        literals, and returning an owned `String` would drag its construction
        into every binary that only ever renders a plan."""
        if self.value == 0:
            return "second"
        elif self.value == 1:
            return "minute"
        elif self.value == 2:
            return "hour"
        elif self.value == 3:
            return "day"
        elif self.value == 4:
            return "month"
        elif self.value == 5:
            return "quarter"
        else:
            return "year"

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def __ne__(self, other: Self) -> Bool:
        return self.value != other.value

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.to_string())


comptime unit_second = CalendarUnit(0)
comptime unit_minute = CalendarUnit(1)
comptime unit_hour = CalendarUnit(2)
comptime unit_day = CalendarUnit(3)
comptime unit_month = CalendarUnit(4)
comptime unit_quarter = CalendarUnit(5)
comptime unit_year = CalendarUnit(6)


struct _Iso8601Text[origin: ImmOrigin]:
    """ISO-8601 timestamp text, read the way Arrow C++'s
    `ParseTimestampISO8601` reads it: every part sits at a fixed offset, so
    each method reads one and answers `None` when it is malformed or out of
    range, and `ticks` composes them.
    """

    var text: Span[Byte, Self.origin]

    def __init__(out self, text: Span[Byte, Self.origin]):
        self.text = text

    @always_inline
    def _at(self, index: Int, char: StaticString) -> Bool:
        return self.text[index] == char.as_bytes()[0]

    @always_inline
    def number(self, start: Int, count: Int) -> Optional[Int]:
        """`count` ASCII digits at `start`, as a number."""
        var value = 0
        for i in range(start, start + count):
            var d = Int(self.text[i]) - ord("0")
            if d < 0 or d > 9:
                return None
            value = value * 10 + d
        return value

    def date(self) -> Optional[CivilDate]:
        """The `YYYY-MM-DD` the text opens with, if it names a real day."""
        if len(self.text) < 10 or not self._at(4, "-") or not self._at(7, "-"):
            return None
        var year = self.number(0, 4)
        var month = self.number(5, 2)
        var day = self.number(8, 2)
        if not (year and month and day):
            return None
        var date = CivilDate(year.value(), month.value(), day.value())
        if not date.is_valid():
            return None
        return date

    def clock(
        self, start: Int, fields: Int, colon: Bool = True
    ) -> Optional[Int]:
        """`hh`, `hh:mm` or `hh:mm:ss` (`fields` = 1, 2, 3) at `start`, in
        seconds. `colon=False` reads `hhmm`, the zone-offset spelling without
        one."""
        var hours = self.number(start, 2)
        if not hours or hours.value() >= 24:
            return None
        var seconds = hours.value() * 3600
        if fields == 1:
            return seconds
        var minutes: Optional[Int]
        if not colon:
            minutes = self.number(start + 2, 2)
        elif self._at(start + 2, ":"):
            minutes = self.number(start + 3, 2)
        else:
            return None
        if not minutes or minutes.value() >= 60:
            return None
        seconds += minutes.value() * 60
        if fields == 2:
            return seconds
        if not self._at(start + 5, ":"):
            return None
        var secs = self.number(start + 6, 2)
        if not secs or secs.value() >= 60:
            return None
        return seconds + secs.value()

    def zone(self, mut end: Int) -> Optional[Int]:
        """The trailing zone offset — `Z`, `[+-]hh`, `[+-]hhmm` or
        `[+-]hh:mm` — in seconds to add for UTC, with `end` moved in front of
        it. No offset is 0; one that does not parse is `None`. Peeled off
        before the clock is read, exactly as Arrow does."""
        var offset: Optional[Int]
        if self._at(end - 1, "Z"):
            end -= 1
            return 0
        elif self._at(end - 3, "+") or self._at(end - 3, "-"):
            end -= 3
            offset = self.clock(end + 1, 1)
        elif self._at(end - 5, "+") or self._at(end - 5, "-"):
            end -= 5
            offset = self.clock(end + 1, 2, colon=False)
        elif (self._at(end - 6, "+") or self._at(end - 6, "-")) and self._at(
            end - 3, ":"
        ):
            end -= 6
            offset = self.clock(end + 1, 2)
        else:
            return 0
        if not offset:
            return None
        # `+01` means an hour ahead of UTC, so UTC is an hour earlier.
        return -offset.value() if self._at(end, "+") else offset.value()

    def ticks[fraction_digits: Int](self) -> Optional[Int]:
        """Ticks since the epoch, `10**fraction_digits` per second."""
        var date = self.date()
        if not date:
            return None
        var seconds = date.value().to_days() * Epoch.SECONDS_PER_DAY
        var subseconds = 0
        var end = len(self.text)
        if end > 10:
            if not (self._at(10, " ") or self._at(10, "T")):
                return None
            var offset = self.zone(end)
            if not offset:
                return None
            var clock: Optional[Int]
            if end == 13:
                clock = self.clock(11, 1)
            elif end == 16:
                clock = self.clock(11, 2)
            elif end == 19 or (end >= 21 and end <= 29):
                clock = self.clock(11, 3)
            else:
                return None
            if not clock:
                return None
            seconds += clock.value() + offset.value()
            if end > 19:
                var given = end - 20
                if not self._at(19, ".") or given > fraction_digits:
                    return None
                var fraction = self.number(20, given)
                if not fraction:
                    return None
                subseconds = fraction.value() * 10 ** (fraction_digits - given)
        comptime limit = Int(Int64.MAX) // 10**fraction_digits
        if seconds > limit or seconds < -limit:
            return None
        # At the top second the fraction itself can carry past the maximum.
        if (
            seconds == limit
            and subseconds > Int(Int64.MAX) - limit * 10**fraction_digits
        ):
            return None
        return seconds * 10**fraction_digits + subseconds


@always_inline
def _write_padded[W: Writer](mut writer: W, value: Int, width: Int):
    """`value` (non-negative) in decimal, left-padded with zeros to `width`
    digits."""
    var digits = 1
    var bound = 10
    while value >= bound:
        digits += 1
        bound *= 10
    for _ in range(width - digits):
        writer.write("0")
    writer.write(value)


struct Iso8601[fraction_digits: Int]:
    """ISO-8601 timestamps as ticks since the epoch, `10**fraction_digits`
    ticks per second — 0, 3, 6 or 9 for second, milli, micro and nano. A
    namespace, never instantiated: `Iso8601[3].parse(text)` and
    `Iso8601[3].write(ticks, writer)`.

    `parse` is a port of Arrow C++'s `ParseTimestampISO8601`, so a string that
    marrow infers as a timestamp is exactly one Arrow would, and `write` is its
    inverse.
    """

    comptime per_second = 10**Self.fraction_digits

    @staticmethod
    def parse(text: Span[Byte, _]) -> Optional[Int]:
        """`text` as ticks, or `None` if it does not parse: `YYYY-MM-DD`,
        optionally followed by `[ T]hh`, `[ T]hh:mm` or `[ T]hh:mm:ss`, then up
        to `fraction_digits` fractional digits after a `.`, then an optional
        zone offset `Z`, `[+-]hh`, `[+-]hhmm` or `[+-]hh:mm`, which is folded
        into the UTC value. A value that does not fit an `Int` in the unit is
        `None`."""
        comptime assert (
            Self.fraction_digits == 0
            or Self.fraction_digits == 3
            or Self.fraction_digits == 6
            or Self.fraction_digits == 9
        ), "fraction_digits must be 0, 3, 6 or 9"
        return _Iso8601Text(text).ticks[Self.fraction_digits]()

    @staticmethod
    def write[W: Writer](ticks: Int, mut writer: W):
        """`ticks` as `YYYY-MM-DD hh:mm:ss`, with the fraction when it is not
        zero: what `parse` reads back to the same value. A second-unit value is
        exactly the text Arrow C++ infers as `timestamp[s]`; a year outside
        0-9999 is written as its plain number, which `parse` rejects."""
        var seconds = floor_div(ticks, Self.per_second)
        var fraction = ticks - seconds * Self.per_second
        var days = floor_div(seconds, Epoch.SECONDS_PER_DAY)
        var clock = seconds - days * Epoch.SECONDS_PER_DAY
        writer.write(CivilDate.from_days(days), " ")
        _write_padded(writer, clock // 3600, 2)
        writer.write(":")
        _write_padded(writer, (clock // 60) % 60, 2)
        writer.write(":")
        _write_padded(writer, clock % 60, 2)
        comptime if Self.fraction_digits > 0:
            if fraction != 0:
                writer.write(".")
                _write_padded(writer, fraction, Self.fraction_digits)


struct WallClock(Copyable, Movable):
    """A time zone and the tick rate of the timestamps read in it."""

    var zone: TimeZone
    var ticks_per_second: Int

    def __init__(out self, var zone: TimeZone, ticks_per_second: Int):
        self.zone = zone^
        self.ticks_per_second = ticks_per_second

    @staticmethod
    def is_utc(name: StringSlice) -> Bool:
        """Whether a timestamp's `timezone` names UTC, whose wall clock is its
        UTC time."""
        return name == "" or name == "UTC" or name == "Etc/UTC" or name == "Z"

    @staticmethod
    def load_zone(name: String) raises -> TimeZone:
        """The zone a timestamp's `timezone` names: UTC, a fixed `+HH:MM`
        offset (the spelling Arrow allows), or an IANA name."""
        if Self.is_utc(name):
            return TimeZone(0, "UTC")
        try:
            if name.startswith("+") or name.startswith("-"):
                return TimeZone.from_utc(name)
            return TimeZone.from_name(name)
        except e:
            raise KeyError(t"tz: unknown time zone '{name}' ({e})")

    @staticmethod
    def of(zone: String, ticks_per_second: Int) raises -> Optional[WallClock]:
        """The clock timestamps in `zone` are read on; None for a naive or UTC
        zone, which needs no conversion."""
        if Self.is_utc(zone):
            return None
        return WallClock(Self.load_zone(zone), ticks_per_second)

    def offset_at(self, utc_seconds: Int) raises -> Int:
        """Seconds east of UTC at the instant `utc_seconds`."""
        try:
            return self.zone.at(utc_seconds).offset
        except e:
            raise IOError(t"tz: {self.zone}: {e}")

    def to_local(self, ticks: Int) raises -> Int:
        """UTC ticks to wall-clock ticks, keeping the sub-second part."""
        var secs = floor_div(ticks, self.ticks_per_second)
        return ticks + self.offset_at(secs) * self.ticks_per_second

    def to_utc(self, ticks: Int) raises -> Int:
        """Wall-clock ticks to UTC ticks, `to_local`'s inverse."""
        var tps = self.ticks_per_second
        var local = floor_div(ticks, tps)
        var days = floor_div(local, 86400)
        var tod = local - days * 86400
        var c = CivilDate.from_days(days)
        var offset: Int
        try:
            offset = self.zone.resolve(
                c.year, c.month, c.day, tod // 3600, (tod // 60) % 60, tod % 60
            ).offset
        except e:
            raise IOError(t"tz: {self.zone}: {e}")
        return ticks - offset * tps


struct TimestampFormat(Copyable, Movable):
    """How one timestamp type is written as text, DuckDB's spelling:
    `2021-06-15 14:30:45`, a fraction only when non-zero and without trailing
    zeros, and for a zoned timestamp its wall-clock time followed by the
    offset — `+02`, or `+05:30` when the offset has minutes."""

    var clock: Optional[WallClock]
    """None for a naive or UTC timestamp, whose wall clock is UTC."""
    var zoned: Bool
    """Whether to write an offset; true for UTC too, which writes `+00`."""
    var ticks_per_second: Int
    var digits: Int
    """Fraction digits one tick needs: 0, 3, 6 or 9."""

    def __init__(out self, zone: String, ticks_per_second: Int) raises:
        """The format of timestamps in `zone` ("" for naive) stored at
        `ticks_per_second`."""
        self.clock = WallClock.of(zone, ticks_per_second)
        self.zoned = Bool(zone)
        self.ticks_per_second = ticks_per_second
        self.digits = 0
        var scale = self.ticks_per_second
        while scale > 1:
            scale //= 10
            self.digits += 1

    def write(self, mut out: String, ticks: Int) raises:
        """Append the timestamp `ticks` to `out`."""
        var tps = self.ticks_per_second
        var secs = floor_div(ticks, tps)
        var frac = ticks - secs * tps
        var offset = self.clock.value().offset_at(secs) if self.clock else 0
        var local = secs + offset
        var days = floor_div(local, 86400)
        var tod = local - days * 86400
        var c = CivilDate.from_days(days)
        Self._pad(out, c.year, 4)
        out += "-"
        Self._pad(out, c.month, 2)
        out += "-"
        Self._pad(out, c.day, 2)
        out += " "
        Self._pad(out, tod // 3600, 2)
        out += ":"
        Self._pad(out, (tod // 60) % 60, 2)
        out += ":"
        Self._pad(out, tod % 60, 2)
        if frac != 0:
            var width = self.digits
            while frac % 10 == 0:
                frac //= 10
                width -= 1
            out += "."
            Self._pad(out, frac, width)
        if self.zoned:
            out += "-" if offset < 0 else "+"
            Self._pad(out, abs(offset) // 3600, 2)
            if abs(offset) % 3600 != 0:
                out += ":"
                Self._pad(out, (abs(offset) // 60) % 60, 2)

    @staticmethod
    def _pad(mut out: String, value: Int, width: Int):
        """`value` in decimal, zero-padded to `width` digits."""
        var digits = String(value)
        for _ in range(width - digits.byte_length()):
            out += "0"
        out += digits
