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
