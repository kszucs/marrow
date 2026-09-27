# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Proleptic-Gregorian calendar arithmetic.

A **leaf module**: it imports nothing from `marrow`, which is the property every
other module under `utils/` has and the reason this one can be shared by
`kernels/temporal.mojo`, `kernels/cast.mojo` and `parquet/reader.mojo` without
any of them depending on each other. Before this existed, the civil-date
algorithms lived inside `kernels/temporal.mojo` as private free functions, and
`parquet/reader.mojo` carried its own epoch constants.

Everything here is integer arithmetic on a **day count** — days since
1970-01-01, the unit every Arrow date and timestamp reduces to. The
*resolution* lookups (`TimeUnit` -> ticks per second, dtype -> nanoseconds per
tick) deliberately stay in `kernels/temporal.mojo`: they are keyed on
`marrow.dtypes` types, and importing those here would cost the leaf property
for two small ladders.

**Shaped for hot loops.** `CivilDate` is three `Int`s with no heap state and no
validity, so it is register-passable and trivially copyable; every method is
`@always_inline`. The extraction kernels call these once per element, so a
non-inlined call or a heap allocation here would show up as a per-row cost.
"""


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

    def __eq__(self, other: Self) -> Bool:
        return (
            self.year == other.year
            and self.month == other.month
            and self.day == other.day
        )

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.year, "-")
        if self.month < 10:
            writer.write("0")
        writer.write(self.month, "-")
        if self.day < 10:
            writer.write("0")
        writer.write(self.day)


struct _Iso8601[mut: Bool, //, origin: Origin[mut=mut]]:
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
        return seconds * 10**fraction_digits + subseconds


def parse_iso8601[fraction_digits: Int](text: Span[Byte, _]) -> Optional[Int]:
    """An ISO-8601 timestamp as ticks since the epoch, `10**fraction_digits`
    ticks per second, or `None` if it does not parse.

    A port of Arrow C++'s `ParseTimestampISO8601`, so a string that marrow
    infers as a timestamp is exactly one Arrow would: `YYYY-MM-DD`, optionally
    followed by `[ T]hh`, `[ T]hh:mm` or `[ T]hh:mm:ss`, then up to
    `fraction_digits` fractional digits after a `.`, then an optional zone
    offset `Z`, `[+-]hh`, `[+-]hhmm` or `[+-]hh:mm`, which is folded into the
    UTC value. `fraction_digits` is 0, 3, 6 or 9 for second, milli, micro and
    nano; a value that does not fit an `Int` in that unit is `None`.

    Parameters:
        fraction_digits: Sub-second digits the unit holds.

    Args:
        text: The candidate text, without quotes.
    """
    comptime assert (
        fraction_digits == 0
        or fraction_digits == 3
        or fraction_digits == 6
        or fraction_digits == 9
    ), "fraction_digits must be 0, 3, 6 or 9"
    return _Iso8601(text).ticks[fraction_digits]()


@always_inline
def _write_padded(mut writer: Some[Writer], value: Int, width: Int):
    """`value` in decimal, left-padded with zeros to `width` digits."""
    var digits = 1
    var bound = 10
    while value >= bound:
        digits += 1
        bound *= 10
    for _ in range(width - digits):
        writer.write("0")
    writer.write(value)


def write_iso8601[fraction_digits: Int](ticks: Int, mut writer: Some[Writer]):
    """Write `ticks` since the epoch, `10**fraction_digits` per second, as
    `YYYY-MM-DD hh:mm:ss` with the fraction when it is not zero.

    The inverse of `parse_iso8601`, so whatever this writes that function reads
    back to the same value; a second-unit value is exactly the text Arrow C++
    infers as `timestamp[s]`. A year outside 0-9999 has no four-digit form and
    is written as its plain number, which `parse_iso8601` rejects.

    Parameters:
        fraction_digits: Sub-second digits the unit holds: 0, 3, 6 or 9.
    """
    comptime per_second = 10**fraction_digits
    var seconds = floor_div(ticks, per_second)
    var fraction = ticks - seconds * per_second
    var days = floor_div(seconds, Epoch.SECONDS_PER_DAY)
    var clock = seconds - days * Epoch.SECONDS_PER_DAY
    var date = CivilDate.from_days(days)
    if date.year >= 0 and date.year <= 9999:
        _write_padded(writer, date.year, 4)
    else:
        writer.write(date.year)
    writer.write("-")
    _write_padded(writer, date.month, 2)
    writer.write("-")
    _write_padded(writer, date.day, 2)
    writer.write(" ")
    _write_padded(writer, clock // 3600, 2)
    writer.write(":")
    _write_padded(writer, (clock // 60) % 60, 2)
    writer.write(":")
    _write_padded(writer, clock % 60, 2)
    comptime if fraction_digits > 0:
        if fraction != 0:
            writer.write(".")
            _write_padded(writer, fraction, fraction_digits)
